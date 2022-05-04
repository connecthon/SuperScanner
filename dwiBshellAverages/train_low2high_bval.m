% Matlab script to predict high-bval dwi averages from low-bval dwi aveages of multishell CHARMED dwi acquisitions from CUBRIC's Connectom 
% based on a forward network
% @PedroLuqueLaguna

clear;

train_subject='06400';
train_subject='14445';
train_subject='53679';
train_subject='68443';
train_subject='80344';




% Inputs
charmed=['SuperScanner/charmed_train/derivatives/dwi/sub-',train_subject,'/314_',train_subject,'_CHARMED_denoisedMPPCA_driftCo_TED_gibbsCorrSubVoxShift.nii.gz'];
bvecs=['SuperScanner/charmed_train/derivatives/dwi/sub-',train_subject,'/bvec.eddy_rotated_bvec'];
bvals=['SuperScanner/charmed_train/sub-',train_subject,'/dwi/314_',train_subject,'_CHARMED_2.bval'];
brain_mask=['SuperScanner/charmed_train/derivatives/dwi/sub-',train_subject,'/nodif_brain_mask.nii.gz'];

% read data
charmed_img=niftiread(charmed);

% Find location of each b-val shell
charmed_bvals=load(bvals);
bvals=unique(charmed_bvals);

% Compute a mean b0 volume
b0s=charmed_img(:,:,:,charmed_bvals==0);
mb0=mean(b0s,4);


% Compute the average signal for each b-value shell
mdwi=zeros([size(mb0),length(bvals)]);
for i=1:length(bvals) 
    bshell=find(charmed_bvals==bvals(i));
    mdwi(:,:,:,i)=mean(charmed_img(:,:,:,bshell),4);
end

% Normalise to the mean b0
nmdwi=mdwi./mb0;

% Flat the data to a dimensional matrix of voxels(rows) by dwi averages for each b-value (columns)
S=reshape(nmdwi,[],size(nmdwi,4));

% Identify voxels within the brain and with plausible diffusion signal
mask_img=niftiread(brain_mask);
good_voxels=(min(S,[],2)>0 & max(S,[],2)==1);
S(mask_img==0,2:end)=0;
S(~good_voxels,2:end)=0;
save(['sub-',train_subject,'_nmdwi.mat'],'S');

% Select only good voxels within the brain  
selected_voxels=mask_img(:) & good_voxels;
S=S(selected_voxels,:);


% Randomly sample 10000 voxels for our training data
training=datasample(1:length(S),10000,'Replace',false);
S_in=S(training,1:5);

% Traing the first network to predict the b-val=4000 shell average
S4000_out=S(training,6);
net4000 = feedforwardnet([10 10 10]);
net4000 = train(net4000, S_in', S4000_out');
save('net4000');

% Test the prediction of the network on the rest of the voxels 
testing=setdiff(1:length(S),training);
Y=S(testing,1:5);
GT=S(testing,6);
Ypred = net4000(Y')';
MSE_4000=norm(GT-Ypred)/length(GT);

% Traing the first 4000 network to predict the b-val=6000 shell average
S6000_out=S(training,7);
net6000 = feedforwardnet([10 10 10]);
net6000 = train(net6000, S_in', S6000_out');
save('net6000');

% Test the prediction of the 6000 network on the rest of the voxels 
testing=setdiff(1:length(S),training);
Y=S(testing,1:5);
GT=S(testing,7);
Ypred = net6000(Y')';
MSE_6000=norm(GT-Ypred)/length(GT);

% Create 4D maps of normalised & averaged dwi values (one volume per bval shell)
niftiwrite(nmdwi,['sub-',train_subject,'_nmdwi']);

% Create 4D maps of the predicted normalised & averaged dwi values
s_pred=reshape(nmdwi,[],size(nmdwi,4));
s_pred(:,6)=net4000(s_pred(:,1:5)');
s_pred(:,7)=net6000(s_pred(:,1:5)');
nmdwi_pred=reshape(s_pred,size(nmdwi));
niftiwrite(nmdwi_pred,['sub-',train_subject,'_nmdwi_pred']);







