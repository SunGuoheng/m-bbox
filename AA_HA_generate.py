
from yolo_HA_AA import YOLO



def get_classes(classes_path):
    with open(classes_path, encoding='utf-8') as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names, len(class_names)

if __name__ == "__main__":
    yolo = YOLO()
    lr = 0.2
    c = 1e+5
    input_shape = [640, 640]

    Cuda = True

    # yolo.AA_('./img/1001.png', c=1e-1, lr=0.2, k=1, epoch=400, type_dis='mean')
    # for lr in [0.01, 0.1, 0.5, 1]:
    #     for epoch in [10,20,30,40]:
    # for k in [1, 3, 5,7,9]:
    #     for c in [0.1,1,5,10,100]:
    #         for i in range(1010, 1015):
    #             yolo.HA("./adv0000/", i, c=c, lr=0.2, k=k, epoch=20, type_dis='mean', iou_threshold=0.5, details_loss=False)
    yolo.HA("./adv0000/", 1011, c=1, lr=0.2, k=10, epoch=40, type_dis='mean', iou_threshold=0.5, details_loss=True)



