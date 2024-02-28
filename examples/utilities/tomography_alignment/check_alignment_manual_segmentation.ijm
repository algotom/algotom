/*
This ImageJ macro is used for calculating the tilt and roll of a parallel-beam tomography system
given projections of a dense sphere acquired in the range of [0, 360] degrees. 
It has been written for tomography alignment at beamline I12-JEEP, Diamond Light Source (DLS).
The macro is for manual segmentation of the sphere.

Author: Nghia Vo (NSLS-2, DLS) and Robert Atwood (DLS).
 */
close("*");
displayw=800;
displayw1=1000;
cropratio=0.25;
minnumfile=12;
maxnumfile=361;
answer=getBoolean("Please choose the path of projections");
if (answer==1){
	path = File.openDialog("Select a file");
  	link = File.getParent(path);
  	listfile = getFileList(link);
  	numfile=listfile.length;
  	if (numfile<minnumfile){
  		exit("Number of projections is too small for good alignment.\n");
  	}
	if (numfile>maxnumfile){
  		exit("Number of projections is too large for fast performance.\n");
  	}  	
  	run("Image Sequence...", "open="+link);
  	id1=getImageID();
  	width=getWidth();
  	height=getHeight();
  	zoom=round(displayw/width*100);
  	run("Set... ", "zoom="+zoom);
	setTool("rectangle");
  	centery=round(height/2);
  	cropheight=cropratio*centery;
  	makeRectangle(0, centery-cropheight, width, cropheight*2);
  	title = "Adjust the crop area";
  	msg = "Please adjust the crop area, then click \"OK\".";
  	waitForUser(title, msg);
  	idtool=toolID;
  	if (idtool!=0){
  		exit("Rectangular selection is required. Please run the macro again!");
  	}
  	getSelectionBounds(xf,yf,widthf,heightf);
  	run("Crop");
  	id1=getImageID(); 
  	width1=getWidth();
  	height1=getHeight();
  	zoom=round(displayw1/width1*100);
  	run("Set... ", "zoom="+zoom);  	
  	answer1=getBoolean("Please choose one flat-field");
  	if (answer1==1){
  		path1 = File.openDialog("Select a file");
  		open(path1);
  		run("Set... ", "zoom="+zoom);
  		id2=getImageID();
  		width2=getWidth();
  		height2=getHeight();
  		if (width2 != width || height2 != height){
  			exit(" The sizes of projections (before crop) and flat field do not match.\n");
  		}  		
  		makeRectangle(xf, yf, widthf, heightf);
  		run("Crop");
  		id2=getImageID();
  		imageCalculator("divide create 32-bit stack", id1, id2);
  		id3=getImageID();
  		selectImage(id1);
  		close();
  		selectImage(id2);
  		close();
  		run("Enhance Contrast", "saturated=0.35");
  		run("Set... ", "zoom="+zoom);  		
  		run("Gaussian Blur...", "sigma=3 stack");
  		run("Threshold...");
		title = "Adjust threshold and convert images to binary mask";
  		msg = "Please adjust the threshold and convert images to binary mask (Apply->Convert to Mask).\nAfter all done, click \"OK\".";
  		waitForUser(title, msg);
  		id4=getImageID();
  		answer2=getBoolean("The sample is a ball?");
  		run("Set Measurements...", "  center bounding redirect=None stack decimal=3");
		run("Analyze Particles...", "size=300-Infinity circularity=0.00-1.00 show=Nothing display clear include stack");	  		
		noelement=nSlices;
		noparticle=nResults;
		if (noelement!=noparticle){
			exit(" There are more than 2 or no round-object in the images. \n Please check and adjust threhold!!!");
		}
		midarray=round((noelement-1)/2);
		midarray1=round(midarray/2);
		midarray2=midarray+midarray1;
		listx = newArray(noelement);
		listy = newArray(noelement);
		for (i=0; i<listx.length; i++){
			if (answer2==1){
 				listx[i]=getResult("XM",i);
 				listy[i]=height-getResult("YM",i);
			}
			if (answer2==0){
				n1=getResult("BY",i);
				n1xl=getResult("BX",i);
				n1xr=n1xl+getResult("Width",i);
				listy[i]=height-n1;
				setSlice(i+1);
         			done=false;
          			for (j=n1xl;j<n1xr&&!done;j++){
          				n2=getPixel(j,n1);
          				if (n2>0){
          					listx[i]=j;
          					done=true;
          				}
          			}				
			}
		}
		if (isOpen("Results")) { 
       			selectWindow("Results"); 
       			run("Close"); 
   		}
   		if (isOpen("Threshold")) { 
       			selectWindow("Threshold"); 
       			run("Close"); 
   		}
		selectImage(id4);
  		close();
		Fit.doFit("y = a*x+b", listx, listy);		
		Fit.plot;		
		roll=atan(Fit.p(0))*180.0/PI;
		print("Roll angle (rotating along the z-axis) = ",roll);
		dx=maxArray(listx)-minArray(listx);
		dx1=abs(listx[midarray1]-listx[midarray2]);
		dx2=abs(listx[midarray1]-listx[midarray2-1]);
		dx3=abs(listx[midarray1]-listx[midarray2+1]);
		mindx=minOf(dx1,minOf(dx2,dx3));
		dy=0.0;
		if (mindx==dx1){
			dy=listy[midarray1]-listy[midarray2];	
		}
		if (mindx==dx2){
			dy=listy[midarray1]-listy[midarray2-1];	
		}
		if (mindx==dx3){
			dy=listy[midarray1]-listy[midarray2+1];	
		}		
		tilt=atan(dy/dx*1.0)*180.0/PI;
		print("Tilt angle (rotating along the x-axis) =  ",tilt);
		showMessage("Angles need to be adjusted (copy available from Log window) ", "Roll = "+roll+"\nTill = "+tilt);
  	}
}
function minArray(array) {
    min=array[0];
    for (i=1; i<lengthOf(array); i++) {
        min=minOf(array[i], min);
    }
    return min;
}
function maxArray(array) {
    max=array[0];
    for (i=1; i<lengthOf(array); i++) {
        max=maxOf(array[i], max);
    }
    return max;
}
