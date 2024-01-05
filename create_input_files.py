from utilsFscocoTest import create_input_files_fs
from utilsMscoco import create_input_files_ms
from utilsFscocoComb import create_input_files_fs_comb

from args import argument_parser

parser = argument_parser()
args = parser.parse_args()

dsName = args.DatasetName
JsonPath =args.JsonPath
ImagePath = args.ImagePath
ImagePath2 = args.ImagePath2
OutputPath = args.OutputPath
CaptionPath = args.CaptionPath
cocoBasePath = args.cocoBasePath
splitPercentage = args.splitPercentage
experiment = args.Experiment
cocoAnnotationPath = args.cocoAnnotationPath

if __name__  == '__main__':
    # Create input files (along with word map)    
    if experiment == 'exp1':
        create_input_files_fs_comb(dataset=dsName,
                        karpathy_json_path=JsonPath,
                        image_folder=ImagePath,
                        image_folder2=ImagePath2,
                        captions_per_image=1,
                        min_word_freq=1,
                        output_folder=OutputPath,
                        caption_folder = CaptionPath,
                        fsCombination = True, 
                        split_percentage = int(splitPercentage),
                        max_len=50)    
    elif experiment == 'exp2':
        create_input_files_fs(dataset=dsName,
                        karpathy_json_path=JsonPath,
                        image_folder=ImagePath,
                        captions_per_image=1,
                        min_word_freq=1,
                        output_folder=OutputPath,
                        caption_folder = CaptionPath,
                        max_len=50)
    elif experiment == 'exp3':
        create_input_files_ms(dataset=dsName,
                        coco_annotation_path=cocoAnnotationPath,
                        image_folder=ImagePath,
                        captions_per_image=1,
                        min_word_freq=1,
                        output_folder=OutputPath,
                        caption_folder = CaptionPath,
                        coco_basePath = cocoBasePath,
                        max_len=50)
       
