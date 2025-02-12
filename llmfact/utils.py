import torch
import numpy as np
import pandas as pd

def IoU(n1, n2):
	"""
	:param n1: 1*N
	:param n2: 1*N
	:return: IoU
	"""
	intersect = np.logical_and(n1, n2)
	union = np.logical_or(n1, n2)
	I = np.count_nonzero(intersect)
	U = np.count_nonzero(union)
	return I / U

def write_layer_txt(filename, model):
	layers = []
	with open(filename, 'w', encoding='utf-8') as f:
		for name, _ in model.named_modules():
			layers.append(name)
		f.write("\n".join(layers))

def correlation_activation(data, alpha=0.1):
	neuron_num = data.shape[1]
	correlation_matrix = torch.corrcoef(data.T)

	select_matrix = correlation_matrix[correlation_matrix > alpha]
	act_num = (select_matrix.sum() - neuron_num) / 2

	total_num = (neuron_num * neuron_num - neuron_num) / 2
	act_rate = act_num / total_num

	print("activation rate is {:.4f}".format(act_rate))

	return correlation_matrix, act_num, act_rate

def flip(row):
	if np.sum(row > 0) < np.sum(row < 0):
		row *= -1
	return row

def thresholding(array):
	array1 = array

	for idx, row in enumerate(array):
		row = flip(row)
		row[row < 0] = 0
		T = np.amax(row) * 0.3
		row[np.abs(row) < T] = 0

		row = row / np.std(row)
		array1[idx, :] = row
	return array1

def apply_mask_and_average(mask, feature_matrix):
	averaged_features = []
	feature_masked = []
	for m in mask:
		masked_features = feature_matrix * m.reshape(1, -1)

		mean_features = np.mean(masked_features, axis=0, keepdims=True)

		averaged_features.append(mean_features)
		feature_masked.append(masked_features)

	return feature_masked, averaged_features

def apply_mask_any(mask, feature_matrix):
	mask = np.any(mask, axis=0)
	masked_features = np.array(feature_matrix * mask)
	return masked_features

def evaluate_iou(img2d, template, verbose=1):
	iou = np.zeros((img2d.shape[0], template.shape[0]))
	for i in range(img2d.shape[0]):
		for j in range(template.shape[0]):
			iou[i, j] = IoU(template[j:j + 1, :], img2d[i:i + 1, :])
	if verbose:
		max_iou = iou.max(axis=0)
		max_index = iou.argmax(axis=0)
		print("\nTemplate\t", end="")
		for i in range(template.shape[0]):
			print("{}\t".format(i+1), end="")
		print("\nIoU\t\t", end="")
		for i in range(template.shape[0]):
			print("{:.4f}\t".format(max_iou[i]), end="")
		print("\nIndex\t\t", end="")
		for i in range(template.shape[0]):
			print("{}\t".format(max_index[i]), end="")
		print()
	return iou

def selfSimilarity(components):
	iou = evaluate_iou(components, components, verbose=0)
	iou = pd.DataFrame(iou)
	return iou

def countSimilarity(iou_total, threshold=0.2):
	iou_max = iou_total.max(axis=0)
	count = np.count_nonzero(iou_total > threshold, axis=0)
	for i in range(iou_max.shape[0]):
		print(f"{iou_max[i]:.4f} \t", end="")
	print()
	for i in range(count.shape[0]):
		print(f"{count[i]} \t", end="")
	print()
	for i in range(count.shape[0]):
		print(f"{i+1} \t", end="")
	return count