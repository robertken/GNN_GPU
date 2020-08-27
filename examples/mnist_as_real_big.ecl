//change to this one for the real stuff:
ImageType := DATA784;  
//ImageType := DATA4;  //test only

mnist_dt := RECORD
      INTEGER1  label;
      ImageType image;
END;

mnist_dt_set := RECORD
			UNSIGNED8 id;
      UNSIGNED1 label;
      SET OF UNSIGNED1 pixel;
END;

//translate image bytes to a SET OF UNSIGNED1 bytes

GetSet(ImageType I) := FUNCTION
  PixRec := {UNSIGNED1 Pixels};
  PixDS  := DATASET(SIZEOF(I),
                    TRANSFORM(PixRec,
                              SELF.Pixels := (>UNSIGNED1<)I[COUNTER]));
  RETURN SET(PixDS,Pixels);
END;

GetSetREAL(ImageType I) := FUNCTION
  PixRec := {REAL4 Pixels};
  PixDS  := DATASET(SIZEOF(I),
                    TRANSFORM(PixRec,
                              SELF.Pixels := (>UNSIGNED1<)I[COUNTER]));
  RETURN SET(PixDS,Pixels);
END;

//change to this one for the real stuff:
train0 := DATASET('~mnist::big::train', mnist_dt, THOR); 
//ds := DATASET([{1,D'1234'}],mnist_dt); //test only

test0 := DATASET('~mnist::big::test', mnist_dt,THOR);

trainDat := PROJECT(train0,
             TRANSFORM(mnist_dt_set,
											 SELF.id := COUNTER,
                       SELF.label := (UNSIGNED1)LEFT.label,
                       SELF.pixel := GetSet(LEFT.image)));
											 			 
testDat:= PROJECT(test0,
					TRANSFORM(mnist_dt_set,
										SELF.id := COUNTER,
										SELF.label := (UNSIGNED1)LEFT.label,
										SELF.pixel := GetSet(LEFT.image)));



trainDatREAL := PROJECT(train0,
             TRANSFORM(mnist_dt_set,
											 SELF.id := COUNTER,
                       SELF.label := (REAL4)LEFT.label,
                       SELF.pixel := GetSetREAL(LEFT.image)));
testDatREAL := PROJECT(test0,
             TRANSFORM(mnist_dt_set,
											 SELF.id := COUNTER,
                       SELF.label := (REAL4)LEFT.label,
                       SELF.pixel := GetSetREAL(LEFT.image)));

//OUTPUT(trainDatREAL);
//OUTPUT(testDatREAL);

EXPORT mnist_as_real_big := MODULE
		EXPORT train := trainDatREAL;//trainDat;
		EXPORT test := testDatREAL;//testDat;
END;


