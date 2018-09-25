(* Simplify equations as much as possible *)

binarySimplify[eq_, vars_] := 
 Module[{rels, gb}, rels = (#^2 - # &) /@ vars;
   gb = GroebnerBasis[Join[{eq /. Equal -> Subtract}, rels], vars];
   Simplify@Thread[Complement[gb, rels] == 0]]

Print /@ Input[]

