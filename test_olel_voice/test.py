from huggingface_hub import snapshot_download
from transformers import AutoModel
path = snapshot_download(repo_id="soynade-research/Oolel-Voices")
model = AutoModel.from_pretrained(path, trust_remote_code=True)

import torchaudio
text = """Su ko nit ñi laajee: Ñan ngay defal say filme? day tontu naan: Europe mooy sama marché, waaye ñimay seetaan, Afrique lañu nekk. Fii ci réewum Europe laa wara ñëw ngir wut xaalis bi may defare samay filme, parce que fi la dooley koom-koom bi nekk. Waaye ñi ma jublu, maanaam ñi ma bëgg jàppale ci seen yewwute jaarale ko ci samay filme, Afrique lañu nekk. Du Sénégal kese, waaye ci Afrique yépp."""

wav = model.generate(text,
                     audio_prompt_path="https://huggingface.co/spaces/soynade-research/Oolel-Voices-Demo/resolve/main/8_1_c.wav",
                     cfg_weight=0.5,
                     exaggeration=0.2,
                     temperature=0.3
                     )

torchaudio.save("oolel_voices_speech.wav", wav, model.sr)