## Refining Visual Perception for Decoration Display: A Self-Enhanced Deep Captioning Model


### Environment
* Apex (optional, for speed-up and fp16 training)
    ```bash
    git clone https://github.com/jackroos/apex
    cd ./apex
    pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./  
    ```
* Other requirements:
    ```bash
    pip install Cython
    pip install -r requirements.txt
    ```
* Compile
    ```bash
    ./scripts/init.sh
    ```


## Training

```
./scripts/nondist_run.sh <task>/train_end2end.py <path_to_cfg> <dir_to_store_checkpoint>
```

## Evaluation
```
python caption/test.py
```