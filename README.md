# EPR-RPC

## Phrase Detection and Alignment

`dataset`: `snli` or `mnli`

`train`: detect & align phrases in the training split

`val`: detect & align phrases in the validation split

`test`: detect & align phrases in the testing split

```
python phrase_detection_and_alignment.py ${dataset} --train --val --test
```

## Model Training

`mode`: `local`, `global` or `concat`

`dataset`: `snli` (default) or `mnli`

`num_epochs`: `int`, default=`3`

`continue`: load checkpoint

```
python train.py --dataset ${dataset} --epoch ${num_epochs} --continue
```

## Evaluation

`mode`: `local`, `global` or `concat`

`dataset`: `snli` (default) or `mnli`

```
python evaluation.py ${mode} --dataset ${dataset}
```