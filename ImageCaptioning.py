import requests
import torch
from PIL import Image
from tqdm import tqdm
from datasetsIc import CaptionDataset
import torchvision.transforms as transforms
device = "cuda" if torch.cuda.is_available() else "cpu"
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer,GPT2Tokenizer, GPT2TokenizerFast, VisionEncoderDecoderModel, AutoImageProcessor, GPT2LMHeadModel, GPT2Config
data_folder = r'C:\Users\Asus\Documents\Surrey\Research Project\CodeAnalysis\Caption Generation\a-PyTorch-Tutorial-to-Image-Captioning-master\a-PyTorch-Tutorial-to-Image-Captioning-master\media\output\Test1'
data_name = 'fscocoSyn_1_cap_per_img_1_min_word_freq'
batch_size = 8
workers = 0
num_epochs = 1 # number of epochs
batch_size = 8 # the size of batches
from torch.utils.tensorboard import SummaryWriter
from transformers import EvalPrediction
from torch.optim import AdamW
import evaluate
from torch.utils.tensorboard import SummaryWriter
from torch.optim import AdamW
from transformers import EvalPrediction
import urllib.parse as parse
import os
from IPython.display import display
from utils import *
import matplotlib.pyplot as plt
import seaborn as sns
import skimage.transform
from utilsComet import *
import comet_ml

comet_ml.init(api_key="7X31f7gn4EtQOyEo69Z0kZtku", project_name="BertViz")
# the decoder model that process the image features and generate the caption text
# decoder_model = "bert-base-uncased"
# decoder_model = "prajjwal1/bert-tiny"
#decoder_model = "gpt2"

encoder_model = "microsoft/swin-base-patch4-window7-224-in22k"
decoder_model = "priyasaravana/modelGPTlm_1"
tokenizer_nm = "priyasaravana/tokenGPT_1"
tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_nm)
image_processor = ViTImageProcessor.from_pretrained(encoder_model)
model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
        encoder_model, decoder_model)

# tokenizer = BertTokenizerFast.from_pretrained(decoder_model)
#model_path = r'C:\Users\Asus\Documents\Surrey\Research Project\CodeAnalysis\Caption Generation\a-PyTorch-Tutorial-to-Image-Captioning-master\a-PyTorch-Tutorial-to-Image-Captioning-master\model\tokenizer'
#model_path = os.path.join(os.getcwd(), 'model\LM')
#tokenizer = GPT2TokenizerFast.from_pretrained(model_path)

task = "VisionEncoderdecoderModel"

#configuration = GPT2Config()
# Initializing a model (with random weights) from the configuration

#configuration = GPT2Config.from_pretrained(model_path, output_hidden_states=False)
#decoder_model = GPT2LMHeadModel.from_pretrained(model_path, config=configuration, local_files_only=True)
# load the image processor

#image_processor = AutoImageProcessor.from_pretrained(encoder_model)
# Fine-tuning your Own Image Captioning Model
## Loading the Model
# the encoder model that process the image and return the image features
# encoder_model = "WinKawaks/vit-small-patch16-224"
# encoder_model = "google/vit-base-patch16-224"
# encoder_model = "google/vit-base-patch16-224-in21k"
# define the optimizer
max_length = 32
# load the rouge and bleu metrics
rouge = evaluate.load("rouge")
bleu = evaluate.load("bleu")
summary_writer = SummaryWriter(log_dir="./image-captioning/tensorboard")

def compute_metrics(eval_pred):
    preds = eval_pred.predictions 
    labels = eval_pred.label_ids
    # decode the predictions and labels
    pred_str = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels_str = tokenizer.batch_decode(labels, skip_special_tokens=True)
    # compute the rouge score
    rouge_result = rouge.compute(predictions=pred_str, references=labels_str)
    # multiply by 100 to get the same scale as the rouge score
    rouge_result = {k: round(v * 100, 4) for k, v in rouge_result.items()}
    # compute the bleu score
    bleu_result = bleu.compute(predictions=pred_str, references=labels_str)
    # get the length of the generated captions
    generation_length = bleu_result["translation_length"]
    return {
            **rouge_result, 
            "bleu": round(bleu_result["bleu"] * 100, 4), 
            "gen_len": bleu_result["translation_length"] / len(preds)
    }

def get_evaluation_metrics(model, data_loader):
    model.eval()
    # define our dataloader
    # number of testing steps
    n_test_steps = len(data_loader)
    # initialize our lists that store the predictions and the labels
    predictions, labels = [], []
    # initialize the test loss
    test_loss = 0.0
    for batch in tqdm(data_loader, "Evaluating"):
        # get the batch
        pixel_values = batch[0]
        label_ids = batch[1]
        # forward pass
        outputs = model(pixel_values=pixel_values, labels=label_ids)
        # outputs = model.generate(pixel_values=pixel_values, max_length=max_length)
        # get the loss
        loss = outputs.loss
        test_loss += loss.item()
        # free the GPU memory
        logits = outputs.logits.detach().cpu()
        # add the predictions to the list
        predictions.extend(logits.argmax(dim=-1).tolist())
        # add the labels to the list
        labels.extend(label_ids.tolist())
    # make the EvalPrediction object that the compute_metrics function expects
    eval_prediction = EvalPrediction(predictions=predictions, label_ids=labels)
    # compute the metrics
    metrics = compute_metrics(eval_prediction)
    # add the test_loss to the metrics
    metrics["test_loss"] = test_loss / n_test_steps
    return metrics

def get_evaluation_metrics(model, dataloader):
    model.eval()
    # define our dataloader
    # dataloader = DataLoader(dataset, collate_fn=collate_fn, batch_size=batch_size)
    # # number of testing steps
    # n_test_steps = len(dataloader)
    # initialize our lists that store the predictions and the labels
    predictions, labels = [], []
    # initialize the test loss
    test_loss = 0.0
    for batch in tqdm(dataloader, "Evaluating"):
        # get the batch
        pixel_values = batch[0]
        label_ids = batch[1]
        # forward pass
        outputs = model(pixel_values=pixel_values, labels=label_ids)
        # outputs = model.generate(pixel_values=pixel_values, max_length=max_length)
        # get the loss
        loss = outputs.loss
        test_loss += loss.item()
        # free the GPU memory
        logits = outputs.logits.detach().cpu()
        # add the predictions to the list
        predictions.extend(logits.argmax(dim=-1).tolist())
        # add the labels to the list
        labels.extend(label_ids.tolist())
    # make the EvalPrediction object that the compute_metrics function expects
    eval_prediction = EvalPrediction(predictions=predictions, label_ids=labels)
    # compute the metrics
    metrics = compute_metrics(eval_prediction)
    # add the test_loss to the metrics
    metrics["test_loss"] = test_loss / 9
    return metrics

# a function to determine whether a string is a URL or not
def is_url(string):
    try:
        result = parse.urlparse(string)
        return all([result.scheme, result.netloc, result.path])
    except:
        return False
    
# a function to load an image
def load_image(image_path):
    if is_url(image_path):
        return Image.open(requests.get(image_path, stream=True).raw)
    elif os.path.exists(image_path):
        return Image.open(image_path)
        

# a function to perform inference
def get_caption(model, image_processor, tokenizer, image_path):
    image = load_image(image_path)
    # preprocess the image
    img = image_processor(image, return_tensors="pt").to(device)
    # generate the caption (using greedy decoding by default)
    output = model.generate(**img)
    # decode the output
    caption = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
    return caption

def show_image_and_captions(url, best_model, tokenizer, image_processor):
    # get the image and display it
    display(load_image(url))
    # get the captions on various models
    our_caption = get_caption(best_model, image_processor, tokenizer, url)
    #finetuned_caption = get_caption(finetuned_model, finetuned_image_processor, finetuned_tokenizer, url)
    #pipeline_caption = get_caption(image_captioner.model, image_processor, tokenizer, url)
    # print the captions
    print(f"Our caption: {our_caption}")
    #print(f"nlpconnect/vit-gpt2-image-captioning caption: {finetuned_caption}")
    #print(f"Abdou/vit-swin-base-224-gpt2-image-captioning caption: {pipeline_caption}")

def main():
    # load the model
    model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
        encoder_model, decoder_model
    ).to(device)

    # initialize the tokenizer
    # tokenizer = AutoTokenizer.from_pretrained(decoder_model)
    
    if "modelGPTlm" in decoder_model:
        # gpt2 does not have decoder_start_token_id and pad_token_id
        # but has bos_token_id and eos_token_id
        tokenizer.pad_token = tokenizer.eos_token # pad_token_id as eos_token_id
        model.config.eos_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id
        # set decoder_start_token_id as bos_token_id
        model.config.decoder_start_token_id = tokenizer.bos_token_id
    else:
        # set the decoder start token id to the CLS token id of the tokenizer
        model.config.decoder_start_token_id = tokenizer.cls_token_id
        # set the pad token id to the pad token id of the tokenizer
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.output_attentions = True

    batch_size = 8
    workers = 0

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    train_loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, data_name, 'TRAIN', transform=transforms.Compose([normalize])),
        batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, data_name, 'VAL', transform=transforms.Compose([normalize])),
        batch_size=1, shuffle=True, num_workers=workers, pin_memory=True)
    # test_loader = torch.utils.data.DataLoader(
    #     CaptionDataset(data_folder, data_name, 'TEST', transform=transforms.Compose([normalize])),
    # batch_size=1, shuffle=True, num_workers=workers, pin_memory=True)
    print(len(train_loader), len(val_loader))
    # using with_transform to preprocess the dataset during training
    # train_loader = train_loader.with_transform(preprocess)
    # val_loader = val_loader.with_transform(preprocess)
    # test_loader  = test_loader.with_transform(preprocess)

    best_checkpoint_number = train(train_loader, val_loader, model)
    #best_checkpoint = 7 # best_checkpoint_number
    
    best_model = VisionEncoderDecoderModel.from_pretrained(f"./image-captioning/checkpoint-{best_checkpoint_number}").to(device)
    best_tokenizer = GPT2TokenizerFast.from_pretrained(f"./image-captioning/checkpoint-{best_checkpoint_number}")
    best_ImageProcessor = ViTImageProcessor.from_pretrained(f"./image-captioning/checkpoint-{best_checkpoint_number }")
    test(best_model, best_tokenizer, best_ImageProcessor)
    show_image_and_captions("http://images.cocodataset.org/test-stuff2017/000000000001.jpg", best_model, best_ImageProcessor, best_tokenizer)  
    # metrics = get_evaluation_metrics(best_model, test_loader)
    # print(f"BLEU: {metrics['bleu']:.4f}, " + 
        # f"ROUGE-1: {metrics['rouge1']:.4f}, ROUGE-2: {metrics['rouge2']:.4f}, ROUGE-L: {metrics['rougeL']:.4f}\n")
    test(test_loader, best_model)

def preprocess(items):
    # preprocess the image
    # raw_image = items[0]
    # normalized_image = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(raw_image)
    pixelValuesLs, targetsLs, image_path = [], [], []
    lenCaptionLoop = len(items[0])
    for i in range(lenCaptionLoop):
        raw_image = items[0][i]        
        resized_img = np.transpose(raw_image, (1, 2, 0))
        resized_array = resized_img.cpu().numpy()
        im = Image.fromarray((resized_array * 255).astype(np.uint8))
        pixel_values = image_processor(im, return_tensors="pt").pixel_values.to(device)
        # tokenize the caption with truncation and padding
        targets = tokenizer([ sentence for sentence in items[1][i] ], 
                            max_length=max_length, padding="max_length", truncation=True, return_tensors="pt").to(device)
        image_path.append(raw_image)
        pixelValuesLs.append(pixel_values)
        targetsLs.append(targets["input_ids"])
    return pixelValuesLs, targetsLs, raw_image

def train(train_loader, val_loader, model): 
    # print some statistics before training
    # number of training steps
    n_train_steps = num_epochs * len(train_loader)
    
    # number of validation steps
    n_valid_steps = len(val_loader)
    # current training step
    current_step = 0
    # logging, eval & save steps
    save_steps = 10
    current_step = 0
    current_step_Train = 0
    current_step_Val = 0
    current_step_count = 0
    save_steps = 10
    best_bleu = 0
    optimizer = AdamW(model.parameters(), lr=1e-5)
    preprocess_transform = transforms.Compose([preprocess])
    
    for epoch in range(num_epochs):
        # set the model to training mode
        model.train()
        # initialize the training loss
        train_loss = 0
        
        for batch in tqdm(train_loader, "Training", total=len(train_loader), leave=False):
            pixel_train_Ls, target_train_Ls, image  = preprocess_transform(batch)
            current_step_Train += 1
            image_path = r'C:\Users\Asus\Desktop\Images\Caroline.jpg'
            print('Started the training: ', current_step_Train)
            for iTrain in range(len(pixel_train_Ls)):
                #print(current_step, save_steps, current_step % save_steps)
                # reset the train and valid loss
                train_loss, valid_loss = 0, 0
                ### training code below ###
                # get the batch & convert to tensor
                pixel_values = pixel_train_Ls[iTrain]
                labels = target_train_Ls[iTrain]
                # forward pass
                outputs = model(pixel_values=pixel_values, labels=labels, output_attentions = True)
                #img = image[iTrain]
                
                attention_weights = outputs.decoder_attentions
                viz_params = {
                    "attention": attention_weights,
                    "default_filter": "all",
                    "bidirectional": False,
                    "display_mode": "light",
                    "layer": None,
                    "head": None,
                }
                experiment.log_asset_data(
                    viz_params,
                    f"attn-view-visionencoder.json",
                )

                experiment.end()
                # t = 0
                # plot the image and attention map
                #fig = plt.figure(figsize=(20, 8))                
                #plt.subplot(int(np.ceil(10 / 5.)), 5, t + 1)
                #plt.text(0, 1, '%s' % ('clock'), color='black', backgroundcolor='white', fontsize=12)
                # plt.imshow(img)
                
                # #alpha = skimage.transform.pyramid_expand(current_alpha, upscale=24, sigma=8)
                # current_alpha_array = attention_weights[0][0][2]
                # # Now you can use current_alpha_array in the pyramid_expand function
                # alpha = skimage.transform.pyramid_expand(current_alpha_array.detach().numpy(), upscale=8, sigma=8)
                # plt.imshow(alpha, alpha=0.4)                
                # plt.axis('off')
                # plt.show()

                # plt.figure(figsize=(8, 6))
                # for i in range(10):
                #     sns.heatmap(attention_weights[0][0][i].detach().numpy(), annot=True, cmap="YlGnBu", fmt=".2f")
                #     plt.xlabel("Input Tokens")
                #     plt.ylabel("Output Tokens")
                #     plt.title("Attention Matrix Heatmap with Attention Weights")
                #     plt.show()
                # get the loss
                loss = outputs.loss
                # backward pass
                loss.backward()
                # update the weights
                optimizer.step()
                # zero the gradients
                optimizer.zero_grad()
                # log the loss
                loss_v = loss.item()
                train_loss += loss_v
                # increment the step
                current_step_count += 1
                current_step += 10
                # log the training loss
                summary_writer.add_scalar("train_loss", loss_v, global_step=current_step)
                iTrain += 1
            
        #if current_step % save_steps == 0:
        ### = code ###
        # evaluate on the validation set
        
        # if the current step is a multiple of the save steps
        #print(f"\nValidation at step {current_step}...\n")
        # set the model to evaluation mode
        model.eval()
        # initialize our lists that store the predictions and the labels
        predictions, labels = [], []
        # initialize the validation loss
        
        valid_loss = 0              

        for batchval in val_loader:
            current_step_Val +=1
            print("Validation at step:", current_step_Val)
            pixel_val_Ls, target_val_Ls  = preprocess_transform(batchval)
            # get the batch
            for iVal in range(len(pixel_val_Ls)):
                pixel_values = pixel_val_Ls[iVal]
                label_ids = target_val_Ls[iVal]
                # forward pass
                outputs = model(pixel_values=pixel_values, labels=label_ids)
                # get the loss
                loss = outputs.loss
                valid_loss += loss.item()
                # free the GPU memory
                logits = outputs.logits.detach().cpu()
                # add the predictions to the list
                predictions.extend(logits.argmax(dim=-1).tolist())
                # add the labels to the list
                labels.extend(label_ids.tolist())
            
        # make the EvalPrediction object that the compute_metrics function expects
        eval_prediction = EvalPrediction(predictions=predictions, label_ids=labels)
        # compute the metrics
        metrics = compute_metrics(eval_prediction)
        # print the stats
        print(f"\nEpoch: {epoch}, Step: {current_step}, Train Loss: {train_loss / save_steps:.4f}, " + 
            f"Valid Loss: {valid_loss / n_valid_steps:.4f}, BLEU: {metrics['bleu']:.4f}, " + 
            f"ROUGE-1: {metrics['rouge1']:.4f}, ROUGE-2: {metrics['rouge2']:.4f}, ROUGE-L: {metrics['rougeL']:.4f}\n")
        # log the metrics

        # recent_bleu = metrics['bleu']
        # is_best = recent_bleu > best_bleu
        # best_bleu = max(recent_bleu, best_bleu)

        summary_writer.add_scalar("valid_loss", valid_loss / n_valid_steps, global_step=current_step)
        summary_writer.add_scalar("bleu", metrics["bleu"], global_step=current_step)
        summary_writer.add_scalar("rouge1", metrics["rouge1"], global_step=current_step)
        summary_writer.add_scalar("rouge2", metrics["rouge2"], global_step=current_step)
        summary_writer.add_scalar("rougeL", metrics["rougeL"], global_step=current_step)
        # save the model
        # if is_best:
        #     best_checkpoint = current_step_count
        model.save_pretrained(f"./image-captioning/checkpoint-{current_step_Val}")
        tokenizer.save_pretrained(f"./image-captioning/checkpoint-{current_step_Val}")
        image_processor.save_pretrained(f"./image-captioning/checkpoint-{current_step_Val}")
        #save_checkpoint(data_name, epoch, 0, encoder_model, decoder_model, optimizer, optimizer, recent_bleu, is_best, data_folder)
        # get the model back to train mode
    return current_step_Val 

def test(best_model, best_tokenizer, best_ImageProcessor):
    # best_model = VisionEncoderDecoderModel.from_pretrained(f"./image-captioning/checkpoint-{best_checkpoint}").to(device)
    # best_tokenizer = GPT2TokenizerFast.from_pretrained(f"./image-captioning/checkpoint-{best_checkpoint}")
    # best_ImageProcessor = ViTImageProcessor.from_pretrained(f"./image-captioning/checkpoint-{best_checkpoint}")
    image_path = r'C:\Users\Asus\Desktop\Images\138259.jpg'
    show_image_and_captions(image_path, best_model, best_tokenizer, best_ImageProcessor)  

if __name__ == '__main__':
    main()  
        

