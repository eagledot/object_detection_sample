import numpy as np
from skimage.transform import resize
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from pathlib import Path
import pandas as pd
import os

import torch
import torch.nn as nn
from  torch.utils.data import Dataset

def resize_image(img,new_shape:tuple,copy=False):

    """Inputs:
        This function implements logic to create a new image with shape as new_shape argument provided keeping the aspect ratio same.
    returns:   np.array of shape new_shape  with same dtype as original image.
    """
    
    new_h,new_w=new_shape
    dtype=img.dtype
    h,w=img.shape[:2]
    scale_h,scale_w=(new_h/h),(new_w/w)
    scale=np.minimum(scale_h,scale_w)
    try:
        if copy==True:
            img=img.copy()
        
        img=resize(img,(int(h*scale),int(w*scale)),preserve_range=True)
        pad_bottom=np.maximum(new_h-img.shape[0],0)
        pad_right=np.maximum(new_w-img.shape[1],0)
        img=np.pad(img,[(0,pad_bottom),(0,pad_right),(0,0)],mode="constant")
        return img.astype(dtype),scale,pad_bottom,pad_right
    except:
        print("scale problem         scale: {}      shape: {}".format(scale,img.shape))
        return np.zeros(shape=(new_h,new_w,3)).astype(dtype)
def std_format(boxes):
    """
        Function to convert the bboxes coordinates from (y1,x1,y2,x2)   to format (center_y,center_x,height,width)  
    Inputs:
            boxes:  shape [N,(y1,x1,y2,x2)]   torch.tensor   
     returns:
            shape:  [N,(center_y,center_x,height,width)]  torch.tensor   
    """
    new=torch.zeros(boxes.shape).float()
    h,w=boxes[:,2]-boxes[:,0],boxes[:,3]-boxes[:,1]
    new[:,:2]=boxes[:,:2]+torch.stack([h/2.,w/2.],dim=1)
    new[:,2:]=torch.stack([h,w],dim=1)   
    
    return new
    
class AnchorGenerator_experimental(nn.Module):
    """
    Arguments:
        sizes (Tuple[Tuple[int]]):
        aspect_ratios (Tuple[Tuple[float]]):
    Returns:   anchors : shape [N,(y1,x1,y2,x2)]  in absolute coordinates.....i.e on the scale of the image_size argument provided in the forward function.
    """

    def __init__(
        self,
        sizes=(128, 256, 512),
        aspect_ratios=(0.5, 1.0, 2.0),
    ):
        super(AnchorGenerator_experimental, self).__init__()
        if not isinstance(sizes[0], (list, tuple)):
            sizes = tuple((s,) for s in sizes)
        if not isinstance(aspect_ratios[0], (list, tuple)):
            aspect_ratios = (aspect_ratios,) * len(sizes)

        assert len(sizes) == len(aspect_ratios)

        self.sizes = sizes
        self.aspect_ratios = aspect_ratios
        self.cell_anchors = None
        self._cache = {}

    @staticmethod
    def generate_anchors(scales, aspect_ratios, device="cpu"):
        scales = torch.as_tensor(scales, dtype=torch.float32, device=device)
        aspect_ratios = torch.as_tensor(aspect_ratios, dtype=torch.float32, device=device)
        h_ratios = torch.sqrt(aspect_ratios)
        w_ratios = 1 / h_ratios

        ws = (w_ratios[:, None] * scales[None, :]).view(-1)
        hs = (h_ratios[:, None] * scales[None, :]).view(-1)

        base_anchors = torch.stack([-ws, -hs, ws, hs], dim=1) / 2
        return base_anchors.round()

    def set_cell_anchors(self, device):
        if self.cell_anchors is not None:
            return self.cell_anchors
        cell_anchors = [
            self.generate_anchors(
                sizes,
                aspect_ratios,
                device
            )
            for sizes, aspect_ratios in zip(self.sizes, self.aspect_ratios)
        ]
        self.cell_anchors = cell_anchors

    def num_anchors_per_location(self):
        return [len(s) * len(a) for s, a in zip(self.sizes, self.aspect_ratios)]

    def grid_anchors(self, grid_sizes, strides):
        anchors = []
        for size, stride, base_anchors in zip(
            grid_sizes, strides, self.cell_anchors
        ):
            grid_height, grid_width = size
            
            print("grid_heigh: {}   grid_width: {}".format(grid_height,grid_width))
            stride_height, stride_width = stride
            print("stride_height: {}    stride_width: {}".format(stride_height,stride_width))
            device = base_anchors.device
            shifts_x = torch.arange(
                0, grid_width, dtype=torch.float32, device=device
            ) * stride_width
            shifts_y = torch.arange(
                0, grid_height, dtype=torch.float32, device=device
            ) * stride_height
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
            shift_x = shift_x.reshape(-1)
            shift_y = shift_y.reshape(-1)
            shifts = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=1)
            
           
            anchors.append(
               (shifts.view(-1, 1, 4) + base_anchors.view(1, -1, 4)).reshape(-1, 4)
            )

        return anchors

    def cached_grid_anchors(self, grid_sizes, strides):
        key = tuple(grid_sizes) + tuple(strides)
        if key in self._cache:
            return self._cache[key]
        anchors = self.grid_anchors(grid_sizes, strides)
        self._cache[key] = anchors
        return anchors

    def forward(self, image_shape, feature_maps_shape):
        assert isinstance(feature_maps_shape,list),"feature_maps_shape must be a list where each element is tuple/list : (batch_size,channels,height,width)"
        assert len(self.sizes)==len(feature_maps_shape)
        
        grid_sizes = tuple([s[-2:] for s in feature_maps_shape])
        image_size = image_shape[-2:]
        
        print("grid_sizes: {}".format(grid_sizes))
        print("image_size: {}".format(image_size))
        
        strides = tuple((image_size[0] / g[0], image_size[1] / g[1]) for g in grid_sizes)
        self.set_cell_anchors("cpu")
        anchors_over_all_feature_maps = self.cached_grid_anchors(grid_sizes, strides)
        anchors = []
        
        for i in range(1):
            anchors_in_image = []
            for anchors_per_feature_map in anchors_over_all_feature_maps:
                anchors_in_image.append(anchors_per_feature_map)
            anchors.append(anchors_in_image)
        anchors = [torch.cat(anchors_per_image) for anchors_per_image in anchors]
        return anchors[0][:,[1,0,3,2]]

def calculate_area(boxes):
    """Calculates the area of N boxes
    
        Input:
         boxes:  [N,(y1,x1,y2,x2)]  torch.tensor
         
        Returns:
         array :  [N,] float
          """
    boxes=boxes.view(-1,4)
    zeros=torch.zeros((boxes.shape[0],)).float()
    
    t1=torch.stack([zeros,boxes[:,2]-boxes[:,0]],dim=1)
    t2=torch.stack([zeros,boxes[:,3]-boxes[:,1]],dim=1)
    
    return (torch.max(t1,dim=-1).values) * (torch.max(t2,dim=-1).values)
    
def calculate_overlap(box_1,boxes):
    
    """Given a box and an array of boxes it calculates the intersection/overlapping  aread of that box_1  with all the boxes
    
    Inputs:
      box_1:  [y1,x1,y2,x2]  array. 
      boxes:   [N,(y1,x1,y2,x2)] array 
    
      Return:
       [N,]  float where each element is the overlapping area .
      
      """
    box_1=box_1.view((-1,4))
    box_1=torch.repeat_interleave(box_1,repeats=boxes.shape[0],dim=0)
    result=torch.zeros(boxes.shape).float()

    for i in range(2):
        
        result[:,i]=torch.max(torch.cat([box_1[:,i:i+1],boxes[:,i:i+1]],dim=-1),dim=-1).values
     
    for i in range(2,4):
        result[:,i]=torch.min(torch.cat([box_1[:,i:i+1],boxes[:,i:i+1]],dim=-1),dim=-1).values
        
    result=calculate_area(result)
    return result
    
def calculate_union(box_1,boxes):
    
    box_1=box_1.view(-1,4)
    box_1=torch.repeat_interleave(box_1,repeats=boxes.shape[0],dim=0)
    
    area_1=calculate_area(box_1)
    area_2=calculate_area(boxes)
    
    return area_1 +area_2

def calculate_IOU(box_1,boxes):
    """calculates Intersection over union for given box with all the other boxes.
    Actually we divide (union-intersection) ..theoritically  this value can never be ZERO ..so we can divide,
    Input:
     box_1:  shape[y1,x1,y2,x2]
     boxes:  shape [N,(y1,x1,y2,x2)]
     
    Return :  [N,]  where each  ith  element is the IOU  b/w box_1 and boxes[i] 
    
    
    """
    intersection=calculate_overlap(box_1,boxes)
    union=calculate_union(box_1,boxes)
    return intersection/(union-intersection)

def generate_rpn_match(anchors,gt_boxes,gt_class_ids,image_shape,ratio):
    """
    Inputs:
        anchors:  [N,4] torch.tensor in (y1,x1,y2,x2)  format.
        gt_boxes:  [n,4] ground-truth bboxes  torch.tensor in (y1,x1,y2,x2)  format.
        ratio:  (0,1)   representing the overlapping ratio..above which an anchor box is considered positive
        gt_class_ids:  [n,] int64 representing the groud-truth class ids for each of the gt_bbox...in range {1..num_classes}
        
    return:
        rpn_match :  shape [N,]  indicating that anchor is positive (1)  ,negative (0) or nothing (-1)...during classification loss...we select only positive and 
                                negative anchor boxes only..
        anchors_class_ids:  long format(int64)  for each of the anchor box telling which class the corresponding anchor box belongs to.
        offsets:  [N,4]  in format  (dy,dx,log(dh),log(dw))  for normal anchor boxes
        idx:      indices indicating the positive anchor boxes...
        We need top 3 and input_image for the training procedure.
    """
    assert len(gt_class_ids)  == gt_boxes.shape[0]
    gt_class_ids=gt_class_ids.view(1,-1)
    IOU=torch.zeros((gt_boxes.shape[0],anchors.shape[0]))
    
    for i  in range(gt_boxes.shape[0]):
        IOU[i]=calculate_IOU(gt_boxes[i],anchors)
    
    IOU=IOU.transpose(1,0)
    IOU_max,IOU_argmax=torch.max(IOU,dim=-1)
    rpn_match=(IOU_max>=ratio).long()
    idx=torch.argmax(IOU,dim=0)
    rpn_match[idx]=1
    
    del idx
    gt_class_ids=gt_class_ids.repeat_interleave(repeats=anchors.shape[0],dim=0)
    anchors_class_ids=rpn_match*gt_class_ids[[i for i in range(anchors.shape[0])],IOU_argmax]
    
    idx=torch.arange(len(rpn_match))[rpn_match==1]
    offsets=torch.zeros((256*2,4)).float()
    pos_anchor_boxes=anchors[idx]
    associated_gt_boxes=torch.stack([gt_boxes[IOU_argmax[i]] for i in idx],dim=0)
    
    temp=generate_offsets(pos_anchor_boxes,associated_gt_boxes,image_shape)
    
    offsets[:len(temp)]=temp
    
    return rpn_match,anchors_class_ids,offsets,idx

def generate_offsets(pos_anchor_boxes,associated_gt_boxes,image_shape:tuple):
    """
    pos_anchor_boxes:  [N,(y1,x1,y2,x2)]  in absolute coordinates
    associated_gt_boxes: [N,(y1,x1,y2,x2)] in absolute coordinates,
    
    image_shape:    (height,width) input_image's shape
    """
    #convert to format (center_y,center_x,height,width)
    height,width=image_shape
   
    pos_anchor_boxes=std_format(pos_anchor_boxes)
    associated_gt_boxes=std_format(associated_gt_boxes)
    
    #convert to normal_coordinates.)
    pos_anchor_boxes=pos_anchor_boxes/torch.tensor([[height,width,height,width]]).float()
    associated_gt_boxes=associated_gt_boxes/torch.tensor([[height,width,height,width]]).float()
    
    #generate_offsets (dy,dx,dh,dw)
    for i in range(2):
        pos_anchor_boxes[:,i]=(associated_gt_boxes[:,i]-pos_anchor_boxes[:,i])/pos_anchor_boxes[:,i+2]
    
    for i in range(2,4):
        pos_anchor_boxes[:,i]=torch.log(associated_gt_boxes[:,i]/pos_anchor_boxes[:,i])
    return pos_anchor_boxes

def cls_predictor(in_channels,num_anchors_per_pixel,num_classes):
    return nn.Conv2d(in_channels,num_classes*num_anchors_per_pixel, kernel_size=3,
                     padding=1)


def bbox_predictor(in_channels,num_anchors):
    #num_anchors is the number of anchors per pixel of that feature map upon which it is being attached
    return nn.Conv2d(in_channels,num_anchors * 4, kernel_size=3, padding=1)

class Config(object):
    ratio=0.5
    MEAN=np.array([0.485, 0.456, 0.406])
    STD = np.array([0.229, 0.224, 0.225])
    
    
    def __init__(self,backbone_network,dense:bool=True):
        self.dummy_tensor=torch.zeros((1,3,self.HEIGHT,self.WIDTH)).float()
        temp=backbone_network
        temp.eval()
        with torch.no_grad():
            
            self.FEATURE_MAPS_SHAPES=[tuple(list(temp_.shape)) for temp_ in  temp(self.dummy_tensor)]
        assert self.HEIGHT in [1024,512,256,128]

        if self.HEIGHT ==1024:
            self.SIZES=[64,128,256,512,1024]
        elif self.HEIGHT==512:
            self.SIZES=[32,64,128,256,512]
        elif self.HEIGHT== 256:
            self.SIZES=[16,32,64,128,256]
        else:
            self.SIZES=[8,16,32,64,128]
        assert len(self.SIZES)  == len(self.FEATURE_MAPS_SHAPES)
        
        if dense:
            scales=[2**0,2**(-1/3),2**(-2/3)]
        else:
            scales=[1.]
        ratios=[0.5,1.,2.]
        
        self.anchor_gen=AnchorGenerator_experimental(sizes=[[base_size*s for s in scales]  for base_size in self.SIZES],aspect_ratios=[ratios]*len(self.SIZES))

        self.ANCHORS=self.anchor_gen((1,3,self.HEIGHT,self.WIDTH),self.FEATURE_MAPS_SHAPES)

        self.ANCHORS=torch.clamp(self.ANCHORS,min=0,max=self.HEIGHT)

        self.NUM_ANCHORS_PER_LOCATION=self.anchor_gen.num_anchors_per_location()
        
        assert self.HEIGHT== self.WIDTH
       
        assert self.ANCHORS.shape[0] == sum([ shape[2]*shape[3]*self.NUM_ANCHORS_PER_LOCATION[i] for i,shape in enumerate(self.FEATURE_MAPS_SHAPES)])
        
        del temp
        
        del self.dummy_tensor

class custom_dataset(Dataset):
    #path is the relative path.
    def __init__(self,config,path,subset=None,aug=False):
        self.aug=aug
        assert subset in ["train","valid"]
        self.config=config
        self.dataset_path=(Path(path)/subset)
        
        
        df=pd.read_csv((self.dataset_path/"labels.txt").as_posix(),header=None)
        
        self.classes=sorted((set(df.iloc[:,5].values)))
        print("No of classes found: {}  {}".format(len(self.classes),self.classes))
        print("Current ratio is {}".format(self.config.ratio))
        
        self.class2id={c:i+1 for i,c in enumerate(self.classes)}
        self.image_ids=list(set(df.iloc[:,0].values))        
        self.image_data={ix:{"bboxes":(df[df.iloc[:,0]==ix].iloc[:,1:5].values).astype("float32")}  for ix in self.image_ids}
        for ix in self.image_ids:
            temp=list(df[df.iloc[:,0]==ix].iloc[:,5].values)
            temp=[self.class2id[t] for t in temp]
            self.image_data[ix]["class_ids"]=np.array(temp).astype("int32")
        
        
        assert sum([1 if ".jpg" in file_name else 0 for file_name in os.listdir(self.dataset_path)])  == len(self.image_ids),"Number of images in the directory is not matching with the images found in labels.txt file"
        
        assert self.config.NUM_CLASSES == len(self.classes),"Number of classes found in given dataset do not match with config object NUM_CLASSES"
        del df
    def preprocess(self,x):
        x=x/255.
        x=x-self.config.MEAN
        x=x/self.config.STD
        return x
    def __len__(self):
        return len(self.image_ids)
    def __getitem__(self,idx):
        assert self.aug == False
        img_id=self.image_ids[idx]
        img=plt.imread((self.dataset_path/img_id).as_posix()).astype("float32")
        img,scale,_,_=resize_image(img,(self.config.HEIGHT,self.config.WIDTH))
        img=self.preprocess(img)
        img=torch.tensor(img).float().permute(2,0,1)
        gt_boxes=torch.tensor(self.image_data[img_id]["bboxes"])*scale
        gt_class_ids=torch.tensor(self.image_data[img_id]["class_ids"]).long()
        rpn_match,ids,offsets,indices=generate_rpn_match(self.config.ANCHORS,gt_boxes,gt_class_ids,(self.config.HEIGHT,self.config.WIDTH),ratio=self.config.ratio)
        return {"img":img,"offsets":offsets,"class_ids":ids,"rpn_match":rpn_match}

def classification_loss(y_true,y_pred,rpn_match):
    """Inputs:
        y_true:  shape [batch_size,num_anchors]  long (int64) format... representing  a class id for each of the anchor
        y_pred:   shape [batch_size,num_anchors,num_classes]    ...logits for each of the anchors for each class
        
        rpn_match:  shape [batch_size,num_anchors]  int64 ...telling about positive ,negative and neutral anchors.
        
        """
    num_classes=y_pred.shape[2]
    bs=y_true.shape[0]
    num_anchors=rpn_match.shape[1]
    y_true=y_true.view(-1)
    y_pred=y_pred.view(-1,num_classes)
    indices=[]
    for i in range(bs):
        temp=rpn_match[i]!=-1  
        idx=torch.arange(start=0,end=len(temp))[temp]
        indices.append(idx+i*num_anchors)
    
    y_true=y_true[torch.cat(indices)]
    y_pred=y_pred[torch.cat(indices)]
    loss_func=nn.CrossEntropyLoss()
    #format is input,target
    loss=loss_func(input=y_pred,target=y_true)
    return loss                  


def regression_loss(y_true,y_pred,rpn_match):
    """Inputs:
        y_true:  shape [batch_size,FIXED_NUMBER,4] ground_truth offsets .
        y_pred:   shape [batch_size,num_anchors,4]   #predicted offsets for each of the anchor
        
        rpn_match:  shape [batch_size,num_anchors]  int64 ...telling about positive ,negative and neutral anchors.
        
        """
    fixed=y_true.shape[1]
    bs=y_true.shape[0]
    num_anchors=rpn_match.shape[1]
    
    y_true=y_true.view(-1,4)
    y_pred=y_pred.view(-1,4)
    indices=[]
    indices_gt=[]
    for i in range(bs):
        temp=rpn_match[i]
        idx=torch.arange(len(temp))[temp==1]
        indices_gt.append(torch.arange(start=0,end=len(idx))+i*fixed)
        indices.append(idx+i*num_anchors)
    y_true=y_true[torch.cat(indices_gt)]
    y_pred=y_pred[torch.cat(indices)]
    
    loss_func=nn.MSELoss()
    loss=loss_func(input=y_pred,target=y_true)
    
    return loss
        