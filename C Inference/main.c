#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <ctype.h>

#define D_EMBD 124
#define VOCAB_SIZE 98
#define MAX_SEQ 256
// Using 8192.0f prevents weight clipping:
#define Q_SCALE 8192.0f 

typedef int16_t q15;

#ifndef WEIGHT_SCALE
#define WEIGHT_SCALE 1.0f
#endif

typedef struct {
    q15 *emb, *pos_emb, *gru_ih, *gru_hh, *gru_bih, *gru_bhh;
    q15 *p_w, *p_b, *m_w, *m_b, *ln_w, *ln_b, *head_w, *head_b;
} Weights;

const char* vocab[] = {
    "\x0a", "\x20", "\x21", "\x22", "\x23", "\x24", "\x26", "\x27", 
    "\x28", "\x29", "\x2a", "\x2b", "\x2c", "\x2d", "\x2e", "\x2f", 
    "\x30", "\x31", "\x32", "\x33", "\x34", "\x35", "\x36", "\x37", 
    "\x38", "\x39", "\x3a", "\x3b", "\x3c", "\x3e", "\x3f", "\x41", 
    "\x42", "\x43", "\x44", "\x45", "\x46", "\x47", "\x48", "\x49", 
    "\x4a", "\x4b", "\x4c", "\x4d", "\x4e", "\x4f", "\x50", "\x51", 
    "\x52", "\x53", "\x54", "\x55", "\x56", "\x57", "\x58", "\x59", 
    "\x5a", "\x61", "\x62", "\x63", "\x64", "\x65", "\x66", "\x67", 
    "\x68", "\x69", "\x6a", "\x6b", "\x6c", "\x6d", "\x6e", "\x6f", 
    "\x70", "\x71", "\x72", "\x73", "\x74", "\x75", "\x76", "\x77", 
    "\x78", "\x79", "\x7a", "\x7c", "\xc2\xad", "\xc2\xb4", "\xc3\xa9", 
    "\xc3\xb1", "\xe2\x80\x8a", "\xe2\x80\x8b", "\xe2\x80\x93", 
    "\xe2\x80\x94", "\xe2\x80\x98", "\xe2\x80\x99", "\xe2\x80\x9c", 
    "\xe2\x80\x9d", "\xe2\x80\xa6", "\x20\xf0\x9f\x8e\x93",
};

const char* anchor_names[] = {
    "Amy","Anna","Ben","Benny","Betsy","Billy","Blue","Bob","Buddy",
    "Clara","Daisy","Emily","Emma","Freddy","Grace","Jack","Jill",
    "Jimmy","Joe","Julie","Leo","Lily","Lilly","Lucy","Max","Mia","Mike",
    "Molly","Mittens","Nick","Pip","Rex","Roxy","Sam","Spot","Sue","Tim",
    "Tom","Tommy","aunt","baby","bear","bird","boy","bunny","cat",
    "chicken","child","cow","dad","deer","dog","doll","dragon",
    "farmer","fish","fox","goose","grandma","grandpa","horse","hippo","kitten",
    "knight","king","lady","man","manor","men","monster","mouse","people","pony",
    "prince","princess","puppy","queen","rabbit","seal","sheep","squirrel","teacher",
    "teddy","tiger","toad","toy","uncle","woman"
};
const int num_anchors = sizeof(anchor_names) / sizeof(anchor_names[0]);

/* -------------------- Utils -------------------- */
static inline float sigmoidf_safe(float x) {
    if (x >= 0.0f) {
        float e = expf(-x);
        return 1.0f / (1.0f + e);
    } else {
        float e = expf(x);
        return e / (1.0f + e);
    }
}

// Higher precision rand float for accurate multinomial sampling:
float rand_f() {
    uint32_t r = (rand() & 0x7FFF) | ((rand() & 0x7FFF) << 15);
    return (float)r / (float)0x3FFFFFFF;
}

int ends_with_name(const char* text, const char* name) {
    int t_len = strlen(text);
    int n_len = strlen(name);
    if (t_len == n_len) {
        for (int i=0; i<n_len; i++) 
            if (tolower(text[i]) != tolower(name[i])) return 0;
        return 1;
    }
    if (t_len > n_len) {
        if (text[t_len - n_len - 1] != ' ') return 0;
        for (int i=0; i<n_len; i++) 
            if (tolower(text[t_len - n_len + i]) != tolower(name[i])) return 0;
        return 1;
    }
    return 0;
}

void get_name_embedding(const char* name, float* out_emb, Weights* w) {
    for (int i=0; i<D_EMBD; i++) out_emb[i] = 0.0f;
    int len = strlen(name);
    for (int c=0; c<len; c++) {
        int idx = 1;
        for (int j=0; j<VOCAB_SIZE; j++) {
            if (strlen(vocab[j])==1 && vocab[j][0] == name[c]) { idx = j; break; }
        }
        for (int i=0; i<D_EMBD; i++) {
            out_emb[i] += (float)w->emb[idx*D_EMBD+i] / Q_SCALE * WEIGHT_SCALE;
        }
    }
    for (int i=0; i<D_EMBD; i++) out_emb[i] /= len;
}

void matvec_f(float* out, const q15* w_q, const float* x_f, const q15* b_q, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        double acc = 0.0;
        if (b_q) acc = (double)b_q[i] / Q_SCALE * WEIGHT_SCALE;
        const q15* row = w_q + (size_t)i * cols;
        for (int j = 0; j < cols; ++j)
            acc += ((double)row[j] / Q_SCALE * WEIGHT_SCALE) * x_f[j];
        out[i] = (float)acc;
    }
}

void load_weights(Weights* w, const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) { printf("Error: %s not found\n", path); exit(1); }

    w->emb = malloc(sizeof(q15)*VOCAB_SIZE*D_EMBD);
    w->pos_emb = malloc(sizeof(q15)*MAX_SEQ*D_EMBD);
    w->gru_ih = malloc(sizeof(q15)*372*(2*D_EMBD));
    w->gru_hh = malloc(sizeof(q15)*372*D_EMBD);
    w->gru_bih = malloc(sizeof(q15)*372);
    w->gru_bhh = malloc(sizeof(q15)*372);
    w->p_w = malloc(sizeof(q15)*D_EMBD);
    w->p_b = malloc(sizeof(q15));
    w->m_w = malloc(sizeof(q15)*D_EMBD*D_EMBD);
    w->m_b = malloc(sizeof(q15)*D_EMBD);
    w->ln_w = malloc(sizeof(q15)*D_EMBD);
    w->ln_b = malloc(sizeof(q15)*D_EMBD);
    w->head_w = malloc(sizeof(q15)*VOCAB_SIZE*D_EMBD);
    w->head_b = malloc(sizeof(q15)*VOCAB_SIZE);

    fread(w->emb, sizeof(q15), VOCAB_SIZE*D_EMBD, f);
    fread(w->pos_emb, sizeof(q15), MAX_SEQ*D_EMBD, f);
    fread(w->gru_ih, sizeof(q15), 372*(2*D_EMBD), f);
    fread(w->gru_hh, sizeof(q15), 372*D_EMBD, f);
    fread(w->gru_bih, sizeof(q15), 372, f);
    fread(w->gru_bhh, sizeof(q15), 372, f);
    fread(w->p_w, sizeof(q15), D_EMBD, f);
    fread(w->p_b, sizeof(q15), 1, f);
    fread(w->m_w, sizeof(q15), D_EMBD*D_EMBD, f);
    fread(w->m_b, sizeof(q15), D_EMBD, f);
    fread(w->ln_w, sizeof(q15), D_EMBD, f);
    fread(w->ln_b, sizeof(q15), D_EMBD, f);
    fread(w->head_w, sizeof(q15), VOCAB_SIZE*D_EMBD, f);
    fread(w->head_b, sizeof(q15), VOCAB_SIZE, f);

    fclose(f);
}

/* STEP */
void step(int idx, float* h, float* m, Weights* w, float* global_context, int use_anchors) {
    
    // The anchor logic is now dynamically injected:
    if (use_anchors) {
        for (int i = 0; i < D_EMBD; i++) {
            m[i] = (0.95f * m[i]) + (0.05f * global_context[i]);
        }
    }

    float in_f[2*D_EMBD];
    for (int i = 0; i < D_EMBD; i++) {
        in_f[i] = (float)w->emb[idx*D_EMBD+i] / Q_SCALE * WEIGHT_SCALE;
        in_f[i+D_EMBD] = m[i]; 
    }

    float i_g[372];
    matvec_f(i_g, w->gru_ih, in_f, w->gru_bih, 372, 2*D_EMBD);

    float h_g[372];
    matvec_f(h_g, w->gru_hh, h, w->gru_bhh, 372, D_EMBD);

    float r[D_EMBD], z[D_EMBD];
    for (int i = 0; i < D_EMBD; i++) {
        r[i] = sigmoidf_safe(i_g[i] + h_g[i]);
        z[i] = sigmoidf_safe(i_g[i + D_EMBD] + h_g[i + D_EMBD]);
    }

    float h_lin[D_EMBD];
    const q15* Uc = w->gru_hh + (size_t)(2*D_EMBD)*D_EMBD;
    const q15* bc = w->gru_bhh + (2*D_EMBD);

    matvec_f(h_lin, Uc, h, bc, D_EMBD, D_EMBD);

    for (int i = 0; i < D_EMBD; i++)
        h_lin[i] = r[i] * h_lin[i];

    float h_new[D_EMBD];
    for (int i = 0; i < D_EMBD; i++) {
        float n = tanhf(i_g[i + 2*D_EMBD] + h_lin[i]);
        h_new[i] = (1.0f - z[i]) * n + z[i] * h[i];
    }

    for (int i = 0; i < D_EMBD; i++)
        h[i] = h_new[i];

    float p_out[1];
    matvec_f(p_out, w->p_w, h, w->p_b, 1, D_EMBD);
    float p = sigmoidf_safe(p_out[0]); 

    float m_cand[D_EMBD];
    matvec_f(m_cand, w->m_w, h, w->m_b, D_EMBD, D_EMBD); 

    for (int i = 0; i < D_EMBD; i++) {
        float cand = tanhf(m_cand[i]);
        m[i] = (1.0f - p) * m[i] + p * cand;
    }
}

/* MAIN */
int main() {
    srand((unsigned)time(NULL));

    Weights w = {0};
    load_weights(&w, "model_q15.bin");

    float h[D_EMBD] = {0};
    float m[D_EMBD] = {0};

    char input[256];
    printf("Enter prompt: ");
    if (!fgets(input, sizeof(input), stdin)) return 0;
    input[strcspn(input, "\r\n")] = 0;

    int len = strlen(input);
    if (len > 0 && input[len - 1] != ' ') {
        input[len] = ' '; input[len + 1] = '\0';
    }

    char gen_text[2048] = {0};
    strcpy(gen_text, input);

    // Initial pass (No anchors during pre-fill):
    for (int t = 0; input[t]; t++) {
        int idx = 1;
        for (int j = 0; j < VOCAB_SIZE; j++) {
            if (strlen(vocab[j]) == 1 && vocab[j][0] == input[t]) {
                idx = j; break;
            }
        }
        step(idx, h, m, &w, NULL, 0);
    }

    printf("\nResponse: %s", input);
    fflush(stdout);

    // Anchor Tracking Variables:
    float character_vault[100][D_EMBD] = {0};
    int matched_names[100] = {0}; 
    int vault_size = 0;
    float global_context[D_EMBD] = {0};

    // Generation Loop:
    for (int k = 0; k < 500; k++) {

        /* LayerNorm */
        double mean = 0.0;
        for (int i = 0; i < D_EMBD; i++) mean += h[i];
        mean /= D_EMBD;

        double var = 0.0;
        for (int i = 0; i < D_EMBD; i++) {
            double d = h[i] - mean; var += d * d;
        }
        var = var / D_EMBD + 1e-5;
        double inv_std = 1.0 / sqrt(var);

        float nh[D_EMBD];
        for (int i = 0; i < D_EMBD; i++) {
            float norm = (float)((h[i] - mean) * inv_std);
            float wln = (float)w.ln_w[i] / Q_SCALE * WEIGHT_SCALE;
            float bln = (float)w.ln_b[i] / Q_SCALE;
            nh[i] = norm * wln + bln;
        }

        /* Logits & Sampling */
        float logits[VOCAB_SIZE];
        matvec_f(logits, w.head_w, nh, w.head_b, VOCAB_SIZE, D_EMBD);

        float mx = logits[0];
        for (int i = 1; i < VOCAB_SIZE; i++)
            if (logits[i] > mx) mx = logits[i];

        float temp = 0.65f;
        double sum = 0.0;
        float probs[VOCAB_SIZE];

        for (int i = 0; i < VOCAB_SIZE; i++) {
            double e = exp((logits[i] - mx) / temp);
            probs[i] = (float)e;
            sum += e;
        }

        for (int i = 0; i < VOCAB_SIZE; i++) probs[i] /= (float)sum;

        float r = rand_f();
        float c = 0.0f;
        int next = 1;

        for (int i = 0; i < VOCAB_SIZE; i++) {
            c += probs[i];
            if (r < c) { next = i; break; }
        }

        printf("%s", vocab[next]);
        fflush(stdout);
        
        if (strcmp(vocab[next], "<") == 0) break;
        
        // Append to tracked text for Anchor matching:
        if (strlen(gen_text) < 2000) strcat(gen_text, vocab[next]);

        // Detect Anchors dynamically:
        for (int n = 0; n < num_anchors; n++) {
            if (ends_with_name(gen_text, anchor_names[n])) {
                if (!matched_names[n]) {
                    matched_names[n] = 1;
                    get_name_embedding(anchor_names[n], character_vault[vault_size], &w);
                    vault_size++;
                }
            }
        }

        int use_anchors = 0;
        if (vault_size > 0) {
            use_anchors = 1;
            for(int i=0; i<D_EMBD; i++) global_context[i] = 0.0f;
            for(int v=0; v<vault_size; v++) {
                for(int i=0; i<D_EMBD; i++) global_context[i] += character_vault[v][i];
            }
            for(int i=0; i<D_EMBD; i++) global_context[i] /= vault_size;
        }

        step(next, h, m, &w, global_context, use_anchors);
    }

    printf("\n");
    return 0;
}
