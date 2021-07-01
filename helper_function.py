import numpy as np

def rect_to_bb(rect):
	# lấy rect từ dlib (left, top, rigth, botttom) = (xmin, ymin, xmax, ymax) 
    # chuyển về dạng (x, y, w, h) hay dùng trong OpenCV
	x = rect.left()
	y = rect.top()
	w = rect.right() - x
	h = rect.bottom() - y
	# trả về tuple
	return (x, y, w, h)


def shape_to_np(shape, dtype="int"):
	# khởi tạo 2d-numpy array
	coords = np.zeros((68, 2), dtype=dtype)		# có 68 điểm
	# duyệt qua 68 facial landmarks và lấy các tọa độ
	for i in range(0, 68):
		coords[i] = (shape.part(i).x, shape.part(i).y)		# lấy tọa độ từng facial landmark
	# trả về 2d-numpy array
	return coords