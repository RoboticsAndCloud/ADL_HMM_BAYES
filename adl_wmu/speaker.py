import simpleaudio as sa

def Speaker(File):
	wavFile = File

	try:
		w_object=sa.WaveObject.from_wave_file(wavFile)
		p_object = w_object.play()
		print("sound is playing")
		p_object.wait_done()
		print("finished")

	except FileNotFoundError:
		print("File not found")

#file='Feedback.wav'
#Speaker(file)
