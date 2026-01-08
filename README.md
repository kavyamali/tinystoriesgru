# tinystoriesgru
A 2.5M parameter TinyStories model trained on GRU with attention, 5x smaller than TinyStories-1M.

The datasheet used here is Tinystories-valid.txt(20MB) downloaded from HuggingFace.

## Core:

The primary reason to train on GRU is the speed efficiency over transformers at very small scale models. Tinystories allowed coherent text generation on models as small as 50MB with tinystories-1M. But is it really possible to achieve more efficiency while retaining the grammar, and long term memory of a full transformer based model?

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

After this optimisation, upon training a 1.5M parameter model, roughly the size of ~6MG over 1000 steps(batch size = 256), the immediate improvements were the grammar context framing. The model was still bad at retrieving almost any long form context at all.

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

But because of the already great efficiency of GRU, the performance is still good and only degrades over very large context generations. (>1000 tokens and so).

The way the attention layer helps is by masking the output with the previous words in a fixed range using standard transformer attention.
The score for any future word is set to -infinity for preventing the model to look in the future.

## Tokeniser:

This model uses a character-level tokeniser instead of using a BPE (Byte Pair Encoding) tokeniser used in GPT-style transformers. The vocab is just all the characters used in the original training datasheet, which allows it to live in the chat.py file itself.
This has allowed for serious efficiency in terms of raw model size, since a 10.5MB model would be ~10MB of pure logic.

## Final model and performance (vs. Tinystories-1M):

The model was trained on an NVIDIA T4 using Google Colab free, in 2 hours and 10 minutes with a batch size of 64 and 4800 steps in total.

Since I don't have the hardware to train this model for any longer, I've included the ```train.py``` for anyone to try.

  ![alt text](https://github.com/kavyamali/tinystoriesgru/blob/main/Graph.png)

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

As it can be seen, the official model (transformer based) is still better with long term context, but is 2x slower on CPU.

The tinystoriesgru model can be directly ran on any machine with python and pytorch by cloning the repository and running ```chat.py```.

# Source:

TinyStories datasheet: https://huggingface.co/datasets/roneneldan/TinyStories
