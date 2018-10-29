# PROGRESS

10%

# USAGE

Edit the file path in `bert.py`

Insert `ret = self.position_embedding(ret)` to the `SymbolModality`'s `bottom_simple` of https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/modalities.py
  
Put `bert.py` to https://github.com/tensorflow/tensor2tensor/tree/master/tensor2tensor/data_generators

Edit https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/data_generators/all_problems.py

Then run https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/bin/t2t_trainer.py

With parameters:
```
--generate_data --problem=bert  --data_dir=../../t2t_data --tmp_dir=../../t2t_data/tmp  --model=transformer_encoder --hparams_set=transformer_tiny --output_dir=../../t2t_train/lm  --train_steps=1000  --eval_steps=100
```

# EXPERIMENT RESULT

|  | next sentence predict |
| ------ | ------ |
| accuracy | 90% |

