import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from datasetsIc import CaptionDataset
from utils import *
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
import torch.nn.functional as F
from tqdm import tqdm
import os
import pickle
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer, GPT2TokenizerFast, VisionEncoderDecoderModel, GPT2TokenizerFast
from IPython.display import display
from transformers import EvalPrediction

# Parameters
data_folder = r'C:\Users\Asus\Documents\Surrey\Research Project\CodeAnalysis\Caption Generation\a-PyTorch-Tutorial-to-Image-Captioning-master\a-PyTorch-Tutorial-to-Image-Captioning-master\media\output\Test'  # folder with data files saved by create_input_files.py
data_name = 'fscocoSyn_1_cap_per_img_1_min_word_freq'  # base name shared by data files
checkpoint = r'C:\Users\Asus\Documents\Surrey\Research Project\CodeAnalysis\Caption Generation\a-PyTorch-Tutorial-to-Image-Captioning-master\a-PyTorch-Tutorial-to-Image-Captioning-master\media\output\Test\checkpoint_fscocoSyn_1_cap_per_img_1_min_word_freq.pth.tar'  # model checkpoint
word_map_file = r'C:\Users\Asus\Documents\Surrey\Research Project\CodeAnalysis\Caption Generation\a-PyTorch-Tutorial-to-Image-Captioning-master\a-PyTorch-Tutorial-to-Image-Captioning-master\media\output\Test\WORDMAP_fscocoSyn_1_cap_per_img_1_min_word_freq.json'  # word map, ensure it's the same the data was encoded with and the model was trained with
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead

print_freq = 10
#curr_dir = os.getcwd()
#data_folder = os.path.join(curr_dir, data_folder)
#checkpoint = os.path.join(curr_dir, checkpoint)
#ord_map_file = os.path.join(curr_dir, word_map_file)
# Load model
# checkpoint = torch.load(checkpoint)
# decoder = checkpoint['decoder']
# decoder.eval()
# encoder = checkpoint['encoder']
# encoder = encoder.to(device)
# encoder.eval()
encoder_model = "microsoft/swin-base-patch4-window7-224-in22k"
decoder_model = "gpt2"
# tokenizer = AutoTokenizer.from_pretrained(decoder_model)
tokenizer = GPT2TokenizerFast.from_pretrained(decoder_model)
# load the image processor
image_processor = ViTImageProcessor.from_pretrained(encoder_model)

def test():
    best_checkpoint = 7
    #Load fine-tuned model
    best_model = VisionEncoderDecoderModel.from_pretrained(f"./image-captioning/checkpoint-{best_checkpoint}").to(device)
    # initialize the tokenizer

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    workers = 0
    test_loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, data_name, 'TEST', transform=transforms.Compose([normalize])),
        batch_size=1, shuffle=True, num_workers=workers, pin_memory=True)
    best_model.eval()  
    test_loss = 0  
    predictions, labels = [], []
    for i, (image, caps, caplens, allcaps) in tqdm(enumerate(test_loader)): 
        pixel_values = image
        label_ids = allcaps       
        #show_image_and_captions(image, best_model, image_processor, tokenizer) 
        outputs = best_model(pixel_values=pixel_values, labels=label_ids)
        # get the loss
        loss = outputs.loss
        test_loss += loss.item()
        # free the GPU memory
        logits = outputs.logits.detach().cpu()
        # add the predictions to the list
        predictions.extend(logits.argmax(dim=-1))
        # add the labels to the list
        labels.extend(label_ids)
        # make the EvalPrediction object that the compute_metrics function expects
        eval_prediction = EvalPrediction(predictions=logits.argmax(dim=-1), label_ids=label_ids)
        metrics = compute_metrics(eval_prediction)
        # print the stats
        print(f"Step: {current_step}, " + 
            f"Test Loss: {Test_loss:.4f}, BLEU: {metrics['bleu']:.4f}, " + 
            f"ROUGE-1: {metrics['rouge1']:.4f}, ROUGE-2: {metrics['rouge2']:.4f}, ROUGE-L: {metrics['rougeL']:.4f}\n")

def compute_metrics(eval_pred):
    preds = eval_pred.predictions 
    labels = eval_pred.label_ids
    # decode the predictions and labels
    #pred_str = tokenizer.batch_decode(preds, skip_special_tokens=True)
    #labels_str = tokenizer.batch_decode(labels, skip_special_tokens=True)

    for example_preds in preds:
        decoded_example = tokenizer.decode(example_preds, skip_special_tokens=True)
        decoded_preds.append(decoded_example)

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

def is_url(string):
    try:
        result = parse.urlparse(string)
        return all([result.scheme, result.netloc, result.path])
    except:
        return False

def load_image(image_path):
    if is_url(image_path):
        return Image.open(requests.get(image_path, stream=True).raw)
    elif os.path.exists(image_path):
        return Image.open(image_path)

# a function to perform imgPrcr
def get_caption(model, image_processor, tokenizer, image):
    #image = load_image(image_path)
    # preprocess the image
    img = image_processor(image, return_tensors="pt").to(device)
    # generate the caption (using greedy decoding by default)
    output = model.generate(**img)
    # decode the output
    caption = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
    return caption
    
def show_image_and_captions(image, best_model, image_processor, tokenizer):
    # get the image and display it
    #display(load_image(url))
    # get the captions on various models
    our_caption = get_caption(best_model, image_processor, tokenizer, image)
    #finetuned_caption = get_caption(finetuned_model, finetuned_image_processor, finetuned_tokenizer, url)
    #pipeline_caption = get_caption(image_captioner.model, image_processor, tokenizer, url)
    # print the captions
    print(f"Our caption: {our_caption}")
    #print(f"nlpconnect/vit-gpt2-image-captioning caption: {finetuned_caption}")
    #print(f"Abdou/vit-swin-base-224-gpt2-image-captioning caption: {pipeline_caption}"

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
    
if __name__ == '__main__':    
    metrics = get_evaluation_metrics(best_model, test_loader)
    test()
    #print("\nBLEU-4 score @ beam size of %d is %.4f." % (beam_size, bleu_score))
