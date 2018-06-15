from face import face_analysis

if __name__ == '__main__':
	video_path = './cut.mp4'
	fa = face_analysis(0)
	age_result, gender_result = fa.run_video(video_path)
	print age_result, gender_result
	fa.free()
