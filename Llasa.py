from transformers import AutoTokenizer, AutoModelForCausalLM
from xcodec2.modeling_xcodec2 import XCodec2Model
import torch
import soundfile as sf

torch.set_default_device('cpu')

llasa_3b ='HKUSTAudio/Llasa-3B'

tokenizer = AutoTokenizer.from_pretrained(llasa_3b)
model = AutoModelForCausalLM.from_pretrained(llasa_3b)
model.eval() 
#model.to('cuda')


 
model_path = "HKUSTAudio/xcodec2"  
 
Codec_model = XCodec2Model.from_pretrained(model_path).cpu()
Codec_model=Codec_model.cpu()
Codec_model.eval()#.cuda()   
# only 16khz speech support!
prompt_wav, sr = sf.read("test.wav")   # you can find wav in Files
prompt_wav = torch.from_numpy(prompt_wav).float().unsqueeze(0)  

prompt_text ="can't. i had someone in my media studies department, my degree, which was associated with the film production."
target_text = 'Hi! So your question for today is umm..., What would you say is your best strenght? I know, it can be a difficult question to anwer.'
#target_text = "Dealing with family secrets is never easy. Yet, sometimes, omission is a form of protection, intending to safeguard some from the harsh truths. One day, I hope you understand the reasons behind my actions. Until then, Anna, please, bear with me."
input_text = prompt_text   + target_text 

def ids_to_speech_tokens(speech_ids):
 
    speech_tokens_str = []
    for speech_id in speech_ids:
        speech_tokens_str.append(f"<|s_{speech_id}|>")
    return speech_tokens_str

def extract_speech_ids(speech_tokens_str):
 
    speech_ids = []
    for token_str in speech_tokens_str:
        if token_str.startswith('<|s_') and token_str.endswith('|>'):
            num_str = token_str[4:-2]

            num = int(num_str)
            speech_ids.append(num)
        else:
            print(f"Unexpected token: {token_str}")
    return speech_ids

#TTS start!
with torch.no_grad():
    # Encode the prompt wav
    vq_code_prompt = Codec_model.encode_code(input_waveform=prompt_wav)
    print("Prompt Vq Code Shape:", vq_code_prompt.shape )   

    vq_code_prompt = vq_code_prompt[0,0,:]
    # Convert int 12345 to token <|s_12345|>
    speech_ids_prefix = ids_to_speech_tokens(vq_code_prompt)

    formatted_text = f"<|TEXT_UNDERSTANDING_START|>{input_text}<|TEXT_UNDERSTANDING_END|>"

    # Tokenize the text and the speech prefix
    chat = [
        {"role": "user", "content": "Convert the text to speech, add laughs and non-verbal sounds:" + formatted_text},
        {"role": "assistant", "content": "<|SPEECH_GENERATION_START|>" + ''.join(speech_ids_prefix)}
    ]

    input_ids = tokenizer.apply_chat_template(
        chat, 
        tokenize=True, 
        return_tensors='pt', 
        continue_final_message=True
    )
    #input_ids = input_ids.to('cuda')
    speech_end_id = tokenizer.convert_tokens_to_ids('<|SPEECH_GENERATION_END|>')

    # Generate the speech autoregressively
    outputs = model.generate(
        input_ids,
        max_length=2048,  # We trained our model with a max length of 2048
        eos_token_id= speech_end_id ,
        do_sample=True,
        top_p=1,           
        temperature=0.8,
    )
    # Extract the speech tokens
    generated_ids = outputs[0][input_ids.shape[1]-len(speech_ids_prefix):-1]

    speech_tokens = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)   

    # Convert  token <|s_23456|> to int 23456 
    speech_tokens = extract_speech_ids(speech_tokens)

    speech_tokens = torch.tensor(speech_tokens).unsqueeze(0).unsqueeze(0)#.cuda()

    # Decode the speech tokens to speech waveform
    print("Generating audio...")
    gen_wav = Codec_model.decode_code(speech_tokens) 


    # if only need the generated part
    gen_wav = gen_wav[:,:,prompt_wav.shape[1]:]
print("Saving generated audio...")
sf.write("gen.wav", gen_wav[0, 0, :].cpu().numpy(), 16000)
