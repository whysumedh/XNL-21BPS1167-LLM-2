![image](https://github.com/user-attachments/assets/dc6a8683-d7bb-4955-a9c6-56e048651e98)

PPT : https://docs.google.com/presentation/d/1rou0hDn2R887swFUCOqYBOyGzLuQZHh_Lfrv_Wbr2o4/edit?usp=sharing

Video and you can find the Final Model checkpoints here too : https://drive.google.com/drive/folders/1B5CuhBQ6FR0IewYcQqt3ZTrbzQ92Cyjx?usp=drive_link

Most of my time went to understanding fine-tuning, researching cloud tiers and using runpod.io remote instance via SSH and connecting to JupyTer and Tensorboard.

Detailed Cost Analysis as asked in Phase 1: https://docs.google.com/document/d/1Ua19HYjceGdUtQm1Mj0bhL0b9mCd3mFzSHvyX3En1Do/edit?usp=sharing

# TensorBoard Results

## TensorBoard Loss/Train Graph For Full Parameter Fine-Tuning
![Screenshot 2025-03-15 022240](https://github.com/user-attachments/assets/42d00e95-f012-454e-89bc-8cc10a5cd57e)

## TensorBoard Loss/Train Graph For Optimized LoRA Fine-Tuning

![Screenshot 2025-03-15 055136](https://github.com/user-attachments/assets/e1387b88-219c-47bb-b8fa-4ca41bb047a2)

Loss of ~3.17 after all steps


![Screenshot 2025-03-15 055244](https://github.com/user-attachments/assets/d2699bf8-ada7-4865-8f68-55a4b0a9c17a)

## Deployment :
![image](https://github.com/user-attachments/assets/ab592fd5-aef8-4761-a362-31f2cfd21e7d)


## Deployment Architecture:
![dLJDKjim4BxxAKGkDJCDAScj9sZeD4mBpSHmPXHhawZ8ArMxfa27Twza1-E4499oOQJlvrkjTtD2B3sNXKcI3-8CPXf1y0B4zt8A7e61kcCZaQylVWYZfrxwX-H0HLwRCmu6Oi7iw7Elv-MVexdcHJaj2NnkaG7vABT5J-MI9ElUxMlpNY69mVUK9ewbd6DkIWKYo0gBGg8If0j](https://github.com/user-attachments/assets/6e220b76-3db4-49c4-a559-85017c08a6a5)

## Enhancements

- Implemented Hyperparameter Tuning with Optuna
- Integrated a Learning Rate Scheduler Using Cosine Annealing

### Results

![2 1423](https://github.com/user-attachments/assets/1705b24f-3a54-4b0a-bf28-35463137815b)






