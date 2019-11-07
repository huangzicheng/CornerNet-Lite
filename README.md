# CornerNet-Lite
training for VOC dataset

The code is an extension of the [CornerNet-lite](https://github.com/princeton-vl/CornerNet-Lite) network dataloader

1.if you want to  train VOC dataset,please add the code in ~/core/db/

2.change ~/config/CornerNet_Squeeze.json

```bash
{
    "system": {
        "dataset": "VOC",
         ...
         "categories": x,  #x is your dataset categories
}
```
3. change ~/core/models/CornerNet_Squeeze.py   in 94~95 rows
```bash
tl_heats = nn.ModuleList([self._pred_mod(X) for _ in range(stacks)])   #x is your dataset categories
br_heats = nn.ModuleList([self._pred_mod(X) for _ in range(stacks)])

```


4.voc.py
```bash
  
    self._voc_cls_ids = [ 1, .....]    #give your ids
    self._voc_cls_names = [ 'person', ....]  # give your labels
    voc_dir = os.path.join('/home/rock/CornerNet-Lite-master/data/', "VOC2012")  # the path of your dataset 
    self._data_dir  = os.path.join(voc_dir, 'JPEGImages')   # training iamge
    self.xml_path = os.path.join(voc_dir, "Annotations")   # the path of xml file
```    
