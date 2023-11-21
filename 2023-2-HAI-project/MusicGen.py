import torch
import torchaudio
import torchaudio.transforms as transforms
from transformers import AutoProcessor, MusicgenForConditionalGeneration

class MusicGen:
	def __init__(self, sample_rate):
		self.processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
		self.model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
		self.prompt = []
		self.audio_values = []
		self.path = 'sample_audio/test.mp3'
		self.sample_rate=sample_rate  # 24000

	def generate_music(self, prompt1, prompt2):
		self.prompt = [prompt1, prompt2]

		inputs = self.processor(
			text=self.prompt,
			padding=True,
			return_tensors="pt",
		)
		
		self.audio_values = self.model.generate(
			**inputs,
   			do_sample=True,
			guidance_scale=3,
			max_new_tokens=256
		)

		self.audio_values = self.audio_values.reshape(2, -1)

		torchaudio.save(
			uri=self.path,
			src=self.audio_values,
			sample_rate=self.sample_rate,
			format='mp3'
		)

	def regenerate_music(self):
		print(f'previous audio shape: {self.audio_values.shape}')
  
		# mono_audio = torchaudio.transforms.DownmixMono(self.audio_values)
		mono_audio = torch.mean(self.audio_values, dim=0, keepdim=True)
		mono_audio = mono_audio.reshape(-1)
		
		print(f'mono audio shape: {mono_audio.shape}')

		inputs = self.processor(
			audio=mono_audio,
			text=self.prompt,
			padding=True,
			return_tensors="pt",
		)

		self.audio_values = self.model.generate(
			**inputs,
   			do_sample=True,
			guidance_scale=3,
			max_new_tokens=256
		)

		self.audio_values = self.audio_values.reshape(2, -1)

		torchaudio.save(
			uri=self.path,
			src=self.audio_values,
			sample_rate=self.sample_rate,
			format='mp3'
		)