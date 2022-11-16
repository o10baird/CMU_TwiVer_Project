File Structure:
data: CNN_RNN/data
- dir for 'test'
- dir for 'train'
src: CNN_RNN/src
- CNN.py
- main.py
- RNN.py
- utils.py

Arguments permitted:
--model, choice from ['CNN', 'RNN'] <-- this is the model to run
--emb, choice from ['Y', 'N'] <-- 'Y' for pretrained emb, 'N' for not
--voc, choice any 'int' <-- this is the total vocabulary
--wandb, choice from ['Y', 'N'] <-- 'Y' for use, 'N' for not

Example run from command line
python main.py --model CNN --emb Y --voc 10000 --wandb N
