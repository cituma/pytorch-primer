import torch

criterion = torch.nn.CrossEntropyLoss()     # 使用交叉熵计算loss, CrossEntropyLoss <==> LogSoftmax + NLLLoss

Y = torch.LongTensor([2, 0, 1])             # ground truth
# 预测值1
Y_pred1 = torch.Tensor([[0.1, 0.2, 0.9],
                        [1.1, 0.1, 0.2],
                        [0.2, 2.1, 0.1]])
# 预测值2
Y_pred2 = torch.Tensor([[0.8, 0.2, 0.3],
                        [0.2, 0.3, 0.5],
                        [0.2, 0.2, 0.5]])

loss1 = criterion(Y_pred1, Y)
loss2 = criterion(Y_pred2, Y)

print("Batch Loss1=", loss1.data, "\nBatch Loss2=", loss2.data)
