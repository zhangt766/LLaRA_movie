
##### Preparation

1. Prepare the environment: 

   ```sh
   git clone https://github.com/zhangt766/LLaRA_Lastfm.git
   cd LLaRA
   pip install -r requirements.txt
   ```

2. Prepare the pre-trained huggingface model of LLaMA2-7B (https://huggingface.co/meta-llama/Llama-2-7b-hf).

3. Download the data (https://huggingface.co/datasets/tongzhang-7/lastfm/tree/main)

4. Prepare the data and checkpoints:

   Put the data to the dir path `data/ref/lastfm` and the checkpoints to the dir path `checkpoints/`.

##### Train LLaRA

Train LLaRA with a single A100 GPU on LastFM dataset:

```sh
sh train_lastfm.sh
```

Note that: set the `llm_path` argument with your own directory path of the Llama2 model.

##### Evaluate LLaRA


Test LLaRA with a single A100 GPU on LastFM dataset:

```sh
sh test_lastfm.sh
```
