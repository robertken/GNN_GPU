mnist_dt := RECORD
	 UNSIGNED1 label;
	 Set of UNSIGNED1 pixel;
END;

ds := RECORD
	UNSIGNED8 id;
	INTEGER1 label;
	DATA784 image;
END;


train := DATASET('~mnist::train', ds, THOR);


OUTPUT(train);