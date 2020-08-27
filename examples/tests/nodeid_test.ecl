IMPORT Std;
ds := DATASET([{'A'},{'B'},{'C'},{'D'},{'E'},
               {'F'},{'G'},{'H'},{'I'},{'J'},
               {'K'},{'L'},{'M'} ,{'N'},{'O'},
               {'P'},{'Q'},{'R'},{'S'},{'T'},
               {'U'},{'V'},{'W'},{'X'},{'Y'},{'Z'}],
              {STRING1 Letter});
dsd := DISTRIBUTE(ds);
pds := PROJECT(dsd,TRANSFORM({STRING1 Letter,UNSIGNED1 node, UNSIGNED1 node2},
                             SELF.node := Std.system.Thorlib.Node()+1, SELF.node2 := Std.system.Thorlib.Nodes(),SELF := LEFT));
OUTPUT(pds,NAMED('Node_Numbered_DS'));



OUTPUT(Std.system.Thorlib.Nodes());

//pds := PROJECT(dsd,TRANSFORM({STRING1 Letter,UNSIGNED1 node},
//                            SELF.node := Std.system.Thorlib.Node()+1, SELF := LEFT));


