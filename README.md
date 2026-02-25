# tinystoriesgru
A 0.2M and 2.5M parameter TinyStories model trained on GRU with attention, much smaller than TinyStories-1M.

The datasheet used here is Tinystories-valid.txt(20MB) downloaded from HuggingFace.

## Core:

GRU uses self-gating through sigmoid.

Instead of standard attention, where
 ,

Sigmoid (1/1+(e)^x), instead, gates in the range (0,1).

Initially, training a pure GRU based model would introduce heavy drifting in logic, with the context retaining entirely on the size of the model, especially with the block(context) size used during training.

The observations with small scale models trained five times and one GRU based model trained with the altered attention logic

Output=σ(Q⋅K/sqr(d))⊙hgru and v, h = self.GRU(x, hidden),

The average context retained overtime was ~50 tokens for a block_size of 500.

That is a 10% loss, which doesn’t scale linearly.

So, it would serve no point training multiple times trying to expect better context retrieval.

The first optimization is to add the output with a saved memory (processed, decoded data), that feeds the hidden state with the memory.
 
This is similar to memory optimisations used for early LSTMs.

For a hidden state at step t (h(t)), M(t-1) would be the memory at the previous step.

For the priority gate ($$p_t = \sigma(W_p h_t + b_p)$$), where sigma is sigmoid, when the value is equal to 1, it overwrites the memory to provide context from previous tokens.

The model creates a proposed memory:

$$
\tilde{M}_t = \tanh(W_c h_t + b_c)
$$

Finally, the old memory is mixed with the new one:

$$
M_t = (1 - p_t) \odot M_{t-1} + p_t \odot \tilde{M}_t
$$


During backpropagation (training), the gradient needs to flow from the end of the story back to the beginning.

In a standard RNN (h = tanh(W*h)), the gradient gets multiplied by $W$ at every step. If $W < 1$, the gradient vanishes (goes to zero).
All of it is, obviously, stored as a cache in the hard drive.

After this optimisation, upon training a 1.5M parameter model, roughly the size of ~6MB over 1000 steps(batch size = 256), the immediate improvements were the grammar context framing. The model was still bad at retrieving almost any long form context at all.

The final optimisation to add was to invite self attention to help for the same.

I used nn.MutliheadAttention to do the same.

In the final model, I used a large GRUcell with a single layer instead of standard GRU with a single or multiple layers.

This made backpropagation easier. The self attention gate lives in an added attn_mask (Attention Mask).

$$
h_t = \text{GRU}([x_t, \text{Context}_t, \text{Memory}_t], h_{t-1})
$$

This makes the GRU computationally much heavier, since:

* Attention: O(T² · d)
* GRUCell: O(d²)
* Total attention cost ≈ Σₜ T² = O(T³)

> This is for full self attention. If more efficient, search query based attention is used, the cost redcuces to O(T²d²). This is demonstrated in tinystoriesgru-0.2M(see ```train.py```).

But because of the already great efficiency of GRU, the performance is still good and only degrades over very large context generations. (>1000 tokens and so).

The way the attention layer helps is by masking the output with the previous words in a fixed range using standard transformer attention.
The score for any future word is set to -infinity for preventing the model to look in the future.

## Tokeniser:

This model uses a character-level tokeniser instead of using a BPE (Byte Pair Encoding) tokeniser used in GPT-style transformers. The vocab is just all the characters used in the original training datasheet, which allows it to live in the chat.py file itself.
This has allowed for serious efficiency in terms of raw model size, since a 10.05MB model would be ~10MB of pure logic.

## Release: tinystoriesgru-0.2M

The 0.2M model is trained with a batch size of 128 and n_embd=96DIMs.The raw FP32 weights are weighed at 1,051KB or ~1MB. 

The model is further quantised to INT8 precision, which is weighed at 271KB.

The 0.2M release introduces the Anchor states in the chat file, which is an attempt to force a fake memory for character names in the datasheet over a span of generation.

This works by introducing a $W_{hh}$ multiplier to the input h(t-1). The eigenvalues are used as a knob to 'fake' the anchor signal.

Using standard tools, the measured spectral radius (highest magnitude of eigenvalue) and Orthogonality were compared.

| Metric                    | FP32 (1MB)      | INT8 (271KB)
| :--- | :--- | :--- |
| Spectral Radius (ρ)       | 1.8842          | 0.5855
| Mean Cosine Similarity    | -0.1900          | -0.1904

## Release: tinystoriespuregru, C Inference

So far, the prominent error noticed in the model has been a $\text{spectral radius} > 1$.

After observation, it seems optimiser (AdamW here) is pushing the wieghts and saturating them to limited dimesntions.

The precise mathematical reason remains unknown; but the most probable guess is the current reccurrence has leaning towards amplification of gain for lower loss. Even an SGD sees similar behaviour.

As the optimiser saturates the sector with the highest/most active eigenvalue, the neurons soon reach the range of the gradient. 

From the four activation gates, we look for ```tanh``` and ```sigmoid```.

Both have a range of $(-1, 1)$. 

Essentially, as these neurons saturate and become flat on the gradient, the loss vibrates. 

The tanh and sigmoid gates act as switches for binary like neurons, as the current step is now equal to the history:

$$h(t) \approx h(t-1)$$

This is for $s(t)$ multiplier is approxiamted to 1. 

The tinystoriespuregru model fixes this, by introducing a spectral leash that limits all four gates to a maximum $\text{eigenvalue}(\max) < 0.95$.

Because the $\text{eigenvalue}(\max) < 1$, the function in exponential form will be contracting, which prevents any explosion.

The model is now ready for a C Inference. 

```quantiseforC.py``` uses a scale of 8192 with int16. This allows the weights to a range of $\pm 4$, which standard clipping prevents.

```main.c``` This is the main function that compiles the .bin using ```gcc```. 

The vocab and weight dimensions are all embedded into the C file using python scripts to measure the model, which are not included here. 

> NOTE: If you train the model on custom datasheet, you muet replace the vocab with your own char list.

The C Inference here is based on a 0.21M parameter model, at 831KB FP32 weights. 

Use ```trainwithoutattention.py``` for the same.

Using standard gcc and UPX(Ultimate Packer for eXecutables), we have ```tiny_infer.exe``` at 15KB.

Thanks to the memory gate/priority bias logic, the model performs exactly the same if not worse as the 271KB and even 10M models, both with attention!

> The anchor logic is now dynamically added in the C inference, see ```main.c``` for the implementation.

For examples regarding the generation, you may refer to the INT8 271KB model below (it's the closest match so far). 

Although it is to be noted, since the complexity is now approximately linear, the generation, even on python, is incredibly fast.

## Final model and performance (vs. Tinystories-1M):

The model was trained on an NVIDIA T4 using Google Colab free, in 2 hours and 10 minutes with a batch size of 64 and 4800 steps in total for the 2.5M model and 1 hour with a bath size of 128 and 10000 steps in total for the 0.2M model.
Since I don't have the hardware to train this model for any longer, I've included the ```train.py``` for anyone to try.

## Graphs:

1) tinystoriesgru

  ![alt text](https://github.com/kavyamali/tinystoriesgru/blob/main/Graphs/Graph.png)

2) tinystoriesgru-0.2M
  
  ![alt text](https://github.com/kavyamali/tinystoriesgru/blob/main/Graphs/Graph-0.2M.png)
  

# Comparision with TinyStories-1M

TinyStories-1M:

| Prompt | Output | Tokens, it/s |
| :--- | :--- | :--- |
| One day, a girl named Sara went to the store. | One day, a girl named Sara went to the store. She saw many things she had never seen before and her mom said "Sara, that is not good."<br><br>Nurse: "Why do you want to buy some of your own things?"<br><br>The little boy replied, "I don't have any friends at home with my toys!"<br><br>Anna was sad but then one day, she noticed something in the mall. It was a big mess inside! Clara's parents were very angry because they got too much money for dinner.<br><br>Clara felt bad for being rude and wanted to help her family. So she asked if she could leave them alone. Her parents told her it no again. After all, Clara learned that sometimes things can be important when we need to find food or vegetables. | 159 tokens, 70.74 it/s |
| Tim found a big blue balloon in the garden. | Tim found a big blue balloon in the garden. He ran to see what was happening and saw that it had been missing out of his hand.<br><br>The little boy said, "I'm sorry because I wanted to help." The small boy tried to open the basket but he couldn't find one. Suddenly, Tim heard a loud roar coming from behind him! He looked around and saw a man with an old beard.<br><br>"Hello there!" shouted the tall man. "What are you doing here?" The young girl asked. "We're going to visit my house soon," replied the man.<br><br>So they set off on their adventure together every day. They were very happy at all for being brave enough home and exploring new places. | 144 tokens, 70.31 it/s |
| The little bird was very sad because he could not fly. | The little bird was very sad because he could not fly.<br><br>The little bird was very sad because he could not fly. He wanted to go on a walk and explore the world around him, but there were no animals who had been in danger.<br><br>One day, Tim saw an animal that lived in the park. It was a big, red cat named Tom. The cat looked at Tim and said, "This is my new friend!"<br><br>Tom felt happy too. But then his mom hugged him tightly and replied, "I'm sorry I didn't know again." She smiled back and said, "It's okay! We can be friends now."<br><br>Tim thanked the kind dog for helping them and they continued playing together. And they all played games together until it was time to go home. | 142 tokens, 69.34 it/s |
| Mom said to Tom, "Please do not touch the hot stove.” | <br><br>Mom said to Tom, "Please do not touch the hot stove."<br><br>But then Lily's mom says: "Lila, you need to go back home and have a nice time. You can play with your toys now."<br><br>"Okay," Ben says. "I will be my friend too. I will wait for me to come soon!"<br><br>They walk around their room and find other things in the kitchen. They see many people that are also there. They run to them and talk and tell stories about animals. They say they are sorry we played with each other. They thank everyone and take turns playing. Lily is happy again. She has made a new house and that she does not like it anymore. She smiles at her mom and says: "It's okay. Now we can help us fix our own food or make them work. We will clean up this mess together." | 170 tokens, 70.86 it/s |
| Once upon a time | Once upon a time, there was a little girl named Lily. She loved to play outside in the sun and pick flowers. Every day she would go on an adventure, even when it came upon something new! It looked very interesting - one special place that way I can find such a beautiful garden with all sorts of amazing things for her friends.<br>Lily knew that she had found lots of fun things to explore and discover wonderful places that were so happy to have seen. Everywhere she went, she saw lots of bright colors of trees and birds flying around them. Finally, while they were out, Lily's mom asked her to come home from school without any steps. | 133 tokens, 67.98 it/s |

tinystoriesgru:

| Prompt | Output | Tokens, it/s |
| :--- | :--- | :--- |
| One day, a girl named Sara went to the store. | One day, a girl named Sara went to the store. She saw a flower on the ground. She thought it was a hard time for her.<br><br>Sarah said, "Thank you for playing. I like pretty rocks and splashing pictures on it. Or maybe we put in your teddy bear on the wall to the park. It is the best cane you have the plane and the soup. Lily and Ben wait for the man's birthday. Sometimes they have a place to the cave and their mom. She says, "The number was in the rain and you can have it back." The little girl was so happy that he had a fun day at the park and | 500 tokens, 123.41 it/s |
| Tim found a big blue balloon in the garden. | Tim found a big blue balloon in the garden. He was so excited to see what was inside the pond.<br><br>"Wow, look at this!" Lily said. She picked up the ball and made a loud noise. The ball became very pretty and cold. The dog was very happy and hugged his mom. They told her that it was not a nice toy. | (253 tokens, 157.83 it/s) |
| The little bird was very sad because he could not fly. | The little bird was very sad because he could not fly. He felt sad and wished he had listened to his mom and dad. He made him feel better and started to cry. His mom said, "I want to spread the blocks to see you." The little boy learned that he was the friendly noise go away and give it back to him. "It is not a toy, he can be a good friend."<br>And they all lived happily ever after. | (330 tokens, 134.20 it/s) |
| Mom said to Tom, "Please do not touch the hot stove." | Mom said to Tom, "Please do not touch the hot stove."<br>The girl was excited to play with the old tree and he was so happy. He was so frightened and he started to cry. His mom laughed and said, "I want to see the coin."<br>The boy hurried off to the living room and saw his friend Tommy for the bear. He was very happy. He ran to the doctor and told his mom that he was not allowed to stay away from the dog.<br>Tom and Lily were sorry and asked, "What is this? That way, you can try and get a chest for you."<br>They started to make noises and see a big box. They | (500 tokens, 119.24 it/s) |
| Once upon a time | Once upon a time there was a baby who lived in a big castle. The bunny was very proud of him and they knew they were going to visit him.<br>As the cow was running and laughing, they said goodbye to the swan. They looked and looked at the man. They learned that it is always difficult to keep it into a patch of light they can be friends and clean up the meadow. | (343 tokens, 151.51 it/s) |

## FP32 and INT8 comparision for tinystoriesgru-0.2M:
 (two examples each, both at temperature 0.5)

## INT8 (271KB):
| Prompt | Output |
| :--- | :--- |
|The little bird was very sad because he could not fly.|The little bird was very sad because he could not fly. They all felt too big and safe. They were sorry for help. Suddenly, a big butterfly with the bark with their friends. They also played with the bird for a pond with her friends. They are happy and safe. He wanted to go on a pond with his mom's car. They were done and said goodbye to the park. They do not like to come back to her. He was so happy and they could help her and said, "I would do not have fun. They saw a big box of the birds. They liked to play with his toys."
| Once upon a time | Once upon a time there was a little boy named Timmy. Timmy was so excited and said, "That's a searce was so beautiful. He wanted to help her finished, and that he was tired and something scared. So, they had to go to the picture from the day, Jimmy was so excited. He was happy that he was very happy to explore the grass. They had a lot of fun that he could not make a swimmer.

## FP32 (1MB):
| Prompt | Output |
| :--- | :--- |
|The little bird was very sad because he could not fly.| The little bird was very sad because he could not fly. They saw that he had found a special bird for her friends. They looked around, but they were so scared. When they were all little girl named Lily and they were so happy. They had so much fun and happy and he could see her. She was so excited to see the birds and even safe. They had to do and she was happy. They looked around and said, "Yes, help you a special cookies. She liked to play with his treat. She was happy that she was very proud of himself and had a fun toys in the sky.
|Once upon a time| Once upon a time there was a little girl and the same scarf toy careful with her mommy to play with his toys. They had a good squirrel and the bird with a special boy. He was so happy that he realized that the ball both had a warm before making a sun with the sea. They all gave her a such a big boy who was going to be so happy. They had a good day she could say sorry and hugged as he held. The bird said that Tom was a long time and said, "I want to play with the big special new pillows and the yummy story."


As it can be seen, the official model (transformer based) is still better with long term context, and faster for large context generation. Note here tokens for transofrmer would also be multiplied with the length per character.

Though the GRU is much faster in shorter bursts (short form generation, as self attention is less of an overhead)

The tinystoriesgru model can be directly ran on any machine with python and pytorch by cloning the repository and running ```chat.py```(for original 2.5M model), ```chat-0.2M.py```(for fp32) and ```chat-0.2M-int8.py```.

# Source:

TinyStories datasheet: https://huggingface.co/datasets/roneneldan/TinyStories

UPX: https://github.com/upx/upx
