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
