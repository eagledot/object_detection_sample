## Object Detection Sample Code:

This implementation is based on the ``SSD`` architecture as described [here](https://arxiv.org/abs/1512.02325)
Since in case of object detection model should be able to localize the object along with recognizing it ,so the deepest features maps which are semantically strong but with poor resolution are not enough to localize the objects with high accuracy due to loss of the localizing information as we go deeper.
So ``SSD`` uses shallower features maps along with deeper feature maps for predicting the object location.
4-5 feature maps are used .

A Separate classification and a Regression subnet is attached to each feature map. i.e no weights are shared .


## Anchors:
Anchors are the fixed bounding boxes which are generated on the whole image and have same spatial dimensions for a particular feature map.
For each feature map anchors base size is directly proportional to the receptive field of that feature map ,Since deeper feature maps have a big receptive field anchors of bigger size are associated. 
We decide the base size and ratios for anchor empirically.


## Loss
### classification loss:
Since for each anchor a class has to be predicted so we resort to categorical based cross-entropy loss.Also since lot of anchors would be belonging to the ``background-class`` this creates an unbalanced class distribution,which should be handled using techniques like ``hard-negative mining`` or ``focal-loss``.

### Regression Loss:
We calculate regression (MSE) loss for only positive anchors,positive anchors are those which have IOU with ground-truth boxes greater than a threshold.For each ground truth bbox ,we would have atleast one anchor associated.
