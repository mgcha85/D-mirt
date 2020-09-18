## Welcome to D-miRT Pages

# Training
## network.py  
training D-miRT model

### class:  
**Generator** -> provide seperated data during training  
**Network** -> training class  
  
### Network methods:  
**sequential_model** (CNN model for sequential model)  
**model** (CNN mode for parallel model)  
**get_file_list** (load and split to test, validation and training data)  
**trainAndValidate** (training parallel model)  
**sequential_train** (training sequential model)  
**crossvalidation** (10-fold cross validation)  
**test_unseen_cell_line** (test the model by using unseen cell line)  


# prediction  
## predict.py  
predict pre-miRNA TSS by using D-miRT model.  
This scans 50kb upstream from the pre-miRNA start.  

### predict methods: 
**get_peak** (get peak data)  
**convert_sequence** (convert strand - to strand +)  
**get_sequence** (get sequence data)  
**set_data** (set both peak and sequence data)  
**run** (predict pre-miRNA TSS)  
**most_likely_tss** (filter the predicted data by two step. first is thresholding (>0.8) and second is to take only duplicates locations)  
**remove_gene_tss** (remove known gene TSS)  
**to_sql** (save predicted data to sql)  


# Evaluation
## evaluation.py
evaluate predicted result by using GRO-cap, H3K4me3 and CAGE-tag .

### evaluation methods: 
**pro** (get GRO-cap feature from PRO-miRNA result)  
**hua** (get GRO-cap feature from HUA et al result)  
**dmirt** (get GRO-cap feature from D-miRT result)  
**h3k4me3** (get H3K4me3 feature from the above three papers' result)  
**cage_tag** (get CAGE-tag feature from the above three papers' result)  


# Visualization
## visualization.py
visualize the trained model by using innvestigate

### visualization methods:   
**visualization** (show trained feature by using innvestigate)  


# Others  
## Database.py  
conrol sqlite data  


## histogram_cl.py
make histogram for training or evaluation data.  
This requires open_cl to use GPU. To use this, you much install open_cl and pyopencl.  

## XmlHandler
save or load XML file  


## fids.xlsx
show that file ID for this training data
## user_param.xml
store user prameter such as step, bandwidth and bin size.


## input and output folder structure
D-miRT requires specific folder sturcture.
The input data and folder structure are linked at here.


