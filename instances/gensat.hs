{-# Language GeneralizedNewtypeDeriving #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE TemplateHaskell #-}

-- code structure based on:
-- https://www.cs.indiana.edu/cgi-pub/sabry/cnf1.hs

import Control.Applicative
import Control.Monad.RWS
import qualified Data.DList as DL
import Data.Foldable (foldrM)
import Data.List (unfoldr, dropWhileEnd, partition, sortBy, transpose)
import Data.List.Split (chunksOf)
import Data.Maybe (fromJust, isNothing)
import qualified Data.Set as Set
import System.Environment (getProgName, getArgs)
import Test.HUnit
import Test.QuickCheck

cnfDebug = False -- show debug comments in the dimacs output
maxThree = False -- make sure every clause has at most three literals
minThree = False -- make sure every clause has at least three literals

-- lsb first
fromBools :: Integral n => [Bool] -> n
fromBools = foldr fromBool 0
  where fromBool False n = 2*n
        fromBool True  n = 2*n + 1

-- lsb first
toBools :: Integral n => n -> [Bool]
toBools = unfoldr binary
  where binary 0 = Nothing
        binary n = Just (r == 1, q)
          where (q, r) = quotRem n 2

prop_toFromBools :: Integral n => NonNegative n -> Bool
prop_toFromBools (NonNegative n) = fromBools (toBools n) == n

prop_fromToBools :: [Bool] -> Bool
prop_fromToBools bs = toBools (fromBools bs) == dropWhileEnd not bs

-- A Bit is either a fixed boolean value or a variable (identified by
-- an integer)
data Bit = Val {getVal :: Bool}
         | Var {getVar :: Int}
         deriving (Eq)

instance Show Bit where
  show (Val False) = "F"
  show (Val True) = "T"
  show (Var v) = show v

isVar, isVal :: Bit -> Bool
isVar (Var _) = True
isVar _       = False
isVal (Val _) = True
isVal _       = False

zero, one:: Bit
zero = Val False
one  = Val True

instance Arbitrary Bit where
  arbitrary = frequency [( 1, liftM Val arbitrary)
                        ,(15, liftM Var (getNonZero <$> arbitrary))]
  shrink (Val b) = [Val b' | b' <- shrink b]
  shrink (Var v) = [Var v' | v' <- shrink v]

fromBits :: Integral n => [Bit] -> Maybe n
fromBits = foldrM fromBit 0
  where fromBit (Var _)     _ = Nothing
        fromBit (Val False) n = Just $ 2 * n
        fromBit (Val True)  n = Just $ 2 * n + 1

toBits :: Integral n => n -> [Bit]
toBits = map Val . toBools

prop_toFromBits :: Integral n => NonNegative n -> Bool
prop_toFromBits (NonNegative n) = (fromJust . fromBits . toBits $ n) == n

prop_fromToBits :: [Bit] -> Property
prop_fromToBits bs =
  if any isVar bs
  then label "trivial" (isNothing (fromBits bs))
  else property $ toBits (fromJust (fromBits bs)) == dropWhileEnd (\b -> b == Val False) bs

padBits :: [Bit] -> [Bit]
padBits = (++ repeat (Val False))

shrinkBits :: [Bit] -> [Bit]
shrinkBits xs = if null ys then [Val False] else ys
  where ys = dropWhileEnd (== Val False) xs

-- zipWith while padding shortest bitstring
zipWithBits :: (Bit -> Bit -> a) -> [Bit] -> [Bit] -> [a]
zipWithBits f xs ys
  | lx < ly = zipWith f px ys
  | lx > ly = zipWith f xs py 
  | otherwise = zipWith f xs ys
  where lx = length xs
        ly = length ys
        px = take ly (padBits xs)
        py = take lx (padBits ys)

zipWithMBits :: Monad m => (Bit -> Bit -> m a) -> [Bit] -> [Bit] -> m [a]
zipWithMBits f xs ys = sequence $ zipWithBits f xs ys

-- negate bit
neg :: Bit -> Bit
neg (Val b) = Val (not b)
neg (Var v) = Var (-v)

prop_doubleNeg :: Bit -> Bool
prop_doubleNeg b = neg (neg b) == b

-- multiplication by 2^n
shift :: Int -> [Bit] -> [Bit]
shift _ [] = []
shift n bs
  | n < 0     = drop (-n) bs
  | otherwise = replicate n (Val False) ++ bs

nextVar :: SymEval Bit
nextVar = state (\v -> (Var v, v+1))

data Clause = Clause { getClause :: [Bit] }
            | Comment { getComment :: String }
            deriving (Eq, Show)

isClause, isComment :: Clause -> Bool
isClause (Clause _) = True
isClause _ = False
isComment (Comment _) = True
isComment _ = False

instance Arbitrary Clause where
  arbitrary = frequency [( 1, liftM Comment arbitrary)
                        ,(15, liftM Clause arbitrary)]
  shrink (Clause c) = [Clause c' | c' <- shrink c]
  shrink (Comment c) = [Comment c' | c' <- shrink c]

newtype Cnf = Cnf { getCnf :: [Clause] }
            deriving (Eq, Show, Arbitrary, Monoid)

removeValues :: Cnf -> Cnf
removeValues = Cnf . map removeFalse . filter removeTrue . getCnf
  where removeTrue (Clause c) = Val True `notElem` c
        removeTrue _ = True
        removeFalse (Clause c) = Clause $ filter (/= Val False) c
        removeFalse c = c

removeComments :: Cnf -> Cnf
removeComments = Cnf . filter isClause . getCnf

toDimacsClauses :: Cnf -> String
toDimacsClauses = unlines . map toDimacsClause . getCnf . removeValues
  where toDimacsClause (Clause c) = unwords . (++["0"]) $ map (show . getVar) c
        toDimacsClause (Comment c) = "c " ++ c

countClauses :: Cnf -> Int
countClauses = length . filter isClause . getCnf

type Desc = Cnf

toDimacs :: Int -> Desc -> String
toDimacs n c = header ++ toDimacsClauses c
  where header = "p cnf " ++ show n ++ " " ++ show (countClauses c) ++ "\n"

seToDimacs :: SymEval a -> String
seToDimacs se = toDimacs (n-1) d
  where (_, n, d) = runRWS se (cnfDebug, maxThree, minThree) 1

addClauses :: MonadWriter Desc m => [[Bit]] -> m ()
addClauses = tell . Cnf . map Clause

debugComment :: (MonadReader (Bool, Bool, Bool) m, MonadWriter Desc m) => String -> m ()
debugComment c = do
  (debug, _, _) <- ask
  tell $ if debug then Cnf [Comment c] else mempty

addComment :: MonadWriter Desc m => String -> m ()
addComment c = tell $ Cnf [Comment c]

-- Symbolic evaluation of the gate/circuit.  The writer logs
-- information about the circuit, while the State can be used
-- to introduce new variables.
type SymEval x = RWS (Bool, Bool, Bool) Desc Int x

eval :: SymEval x -> x
eval se = fst $ evalRWS se (cnfDebug, maxThree, minThree) 1

notGate :: Bit -> SymEval Bit
notGate = return . neg

nandGate :: Bit -> Bit -> SymEval Bit
nandGate (Var x)     (Val y) = nandGate (Val y) (Var x)
nandGate (Val False) y       = return (Val True)
nandGate (Val True)  y       = return (neg y)
nandGate x           y       = do
  z <- nextVar
  (_, _, minThree) <- ask
  if minThree
    then addClauses [[x, z, z]
                    ,[y, z, z]
                    ,[neg x, neg y, neg z]]
    else addClauses [[x, z]
                    ,[y, z]
                    ,[neg x, neg y, neg z]]
  return z

norGate :: Bit -> Bit -> SymEval Bit
norGate x y = nandGate (neg x) (neg y) >>= notGate

andGate :: Bit -> Bit -> SymEval Bit
andGate x y = nandGate x y >>= notGate

orGate :: Bit -> Bit -> SymEval Bit
orGate x y = nandGate (neg x) (neg y)

xorGate :: Bit -> Bit -> SymEval Bit
xorGate (Var x)     (Val y) = xorGate (Val y) (Var x)
xorGate (Val False) y       = return y
xorGate (Val True)  y       = return $ neg y
xorGate x           y       = do
  z <- nextVar
  addClauses [[    x,     y, neg z]
             ,[    x, neg y,     z]
             ,[neg x,     y,     z]
             ,[neg x, neg y, neg z]]
  return z

xnorGate :: Bit -> Bit -> SymEval Bit
xnorGate x y = xorGate x y >>= notGate

-- multiplexer: depending on s, output x or y
muxGate :: Bit -> Bit -> Bit -> SymEval Bit
muxGate (Val False) x _ = return x
muxGate (Val True)  _ y = return y
muxGate s           x y = do
  z <- nextVar
  addClauses [[    s,     x, neg z]
             ,[    s, neg x,     z]
             ,[neg s,     y, neg z]
             ,[neg s, neg y,     z]]
  return z

-- n-bit multiplexer: depending on s, output xs or ys
muxNGate :: Bit -> [Bit] -> [Bit] -> SymEval [Bit]
muxNGate (Val False) xs _  = return xs
muxNGate (Val True)  _  ys = return ys
muxNGate s xs ys = zipWithMBits (muxGate s) xs ys

-- n-bit or-gate: check if any of the bits is True
orNGate :: [Bit] -> SymEval Bit
orNGate []  = return (Val False)
orNGate [x] = return x
orNGate xs  = do
  y <- orNGate ys
  z <- orNGate zs
  orGate y z
  where (ys, zs) = splitAt (length xs `div` 2) xs

-- n-bit or-gate without output
assertOrNGate :: [Bit] -> SymEval ()
assertOrNGate xs = do
  (_, maxThree, minThree) <- ask
  if maxThree || minThree
    then orNGate xs >> return ()
    else addClauses [xs]

-- n-bit and-gate: check if all bits are True
andNGate :: [Bit] -> SymEval Bit
andNGate []  = return (Val True)
andNGate [x] = return x
andNGate xs  = do
  y <- andNGate ys
  z <- andNGate zs
  andGate y z
  where (ys, zs) = splitAt (length xs `div` 2) xs
       
-- n-bit eq-gate: check if all bits in xs are equal to all ys
eqNGate :: [Bit] -> [Bit] -> SymEval Bit
eqNGate xs ys = do
  debugComment $ show xs ++ " == " ++ show ys
  zs <- zipWithMBits xnorGate xs ys
  andNGate zs

assertEqGate :: Bit -> Bit -> SymEval ()
assertEqGate (Val x)     (Val y) = unless (x == y) $ error "assertEqGate"
assertEqGate (Var x)     (Val y) = assertEqGate (Val y) (Var x)
assertEqGate (Val False) y       = do
  debugComment $ show y ++ " == False"
  (_, _, minThree) <- ask
  if minThree
    then addClauses [[neg y, neg y, neg y]]
    else addClauses [[neg y]]
assertEqGate (Val True ) y       = do
  debugComment $ show y ++ " == True"
  (_, _, minThree) <- ask
  if minThree
    then addClauses [[y, y, y]]
    else addClauses [[y]]
assertEqGate x           y       = do
  debugComment $ show x ++ " == " ++ show y
  (_, _, minThree) <- ask
  if minThree
    then addClauses [[x, neg y, neg y]
                    ,[neg x, y, y]]
    else addClauses [[x, neg y]
                    ,[neg x, y]]
  addClauses [[x, neg y], [neg x, y]]

assertFalseGate, assertTrueGate :: Bit -> SymEval ()
assertFalseGate = assertEqGate (Val False)
assertTrueGate = assertEqGate (Val True)

-- 1-bit half adder
halfAdd :: Bit -> Bit -> SymEval (Bit, Bit)
halfAdd x y = do
  debugComment $ "Start HA: " ++ show (x, y)
  s <- xorGate x y
  c <- andGate x y
  debugComment $ "End HA: " ++ show (x, y) ++ " -> " ++ show (s, c)
  return (s, c)

-- test the entire truth table
test_halfAdd = TestList
  [ TestCase . assertEqual "ha-ff" (zero, zero) . eval $ halfAdd zero zero
  , TestCase . assertEqual "ha-ft" (one , zero) . eval $ halfAdd zero one 
  , TestCase . assertEqual "ha-tf" (one , zero) . eval $ halfAdd one  zero
  , TestCase . assertEqual "ha-tt" (zero, one ) . eval $ halfAdd one  one 
  ]

-- 1-bit half adder that assumes carry input one
halfSub :: Bit -> Bit -> SymEval (Bit, Bit)
halfSub x y = do
  debugComment $ "Start HS: " ++ show (x, y)
  s <- xnorGate x y
  c <- orGate x y
  debugComment $ "End HS: " ++ show (x, y) ++ " -> " ++ show (s, c)
  return (s, c)

fullAdd :: Bit -> Bit -> Bit -> SymEval (Bit, Bit)

fullAdd (Val x) (Val y) (Val ci) = return (Val s, Val co)
  where s = (x /= y) /= ci
        co = if ci then x || y else x && y

fullAdd (Val x) (Val y) ci = fullAdd ci (Val x) (Val y)
fullAdd (Val x) y ci = fullAdd y (Val x) ci
fullAdd x (Val y) (Val ci) = return (s, co)
  where s = if y == ci then x else neg x
        co = if y == ci then (Val y) else s

fullAdd x (Val y) ci = fullAdd x ci (Val y)
fullAdd x y (Val ci) = if ci then halfSub x y else halfAdd x y

fullAdd x y ci = do
  (_, maxThree, _) <- ask
  debugComment $ "Start FA: " ++ show (x, y, ci)
  if maxThree
    then do (s1, c1) <- halfAdd x y
            (s2, c2) <- halfAdd s1 ci
            co <- orGate c1 c2
            return (s2, co)
    else do s <- nextVar
            addClauses [[neg x, neg y, neg ci,     s]
                       ,[neg x, neg y,     ci, neg s]
                       ,[neg x,     y, neg ci, neg s]
                       ,[neg x,     y,     ci,     s]
                       ,[    x, neg y, neg ci, neg s]
                       ,[    x, neg y,     ci,     s]
                       ,[    x,     y, neg ci,     s]
                       ,[    x,     y,     ci, neg s]]
            co <- nextVar
            addClauses [[neg x, neg y,             co]
                       ,[neg x,        neg ci,     co]
                       ,[    x,     y,         neg co]
                       ,[    x,            ci, neg co]
                       ,[       neg y, neg ci,     co]
                       ,[           y,     ci, neg co]]
            debugComment $ "End FA: " ++ show (x, y, ci) ++ " -> " ++ show (s, co)
            return (s, co)

-- test the entire truth table
test_fullAdd = TestList
  [ TestCase . assertEqual "fa-fff" (zero, zero) . eval $ fullAdd zero zero zero
  , TestCase . assertEqual "fa-fft" (one , zero) . eval $ fullAdd zero zero one 
  , TestCase . assertEqual "fa-ftf" (one , zero) . eval $ fullAdd zero one  zero
  , TestCase . assertEqual "fa-ftt" (zero, one ) . eval $ fullAdd zero one  one 
  , TestCase . assertEqual "fa-tff" (one , zero) . eval $ fullAdd one  zero zero
  , TestCase . assertEqual "fa-tft" (zero, one ) . eval $ fullAdd one  zero one 
  , TestCase . assertEqual "fa-ttf" (zero, one ) . eval $ fullAdd one  one  zero
  , TestCase . assertEqual "fa-ttt" (one , one ) . eval $ fullAdd one  one  one 
  ]

type Adder = [Bit] -> [Bit] -> SymEval [Bit]
type Subtractor = [Bit] -> [Bit] -> SymEval ([Bit], Bit)

-- rippleAdd with carry in
rippleAddCarry :: [Bit] -> [Bit] -> Bit -> SymEval [Bit]
rippleAddCarry [] [] ci = return [ci]
rippleAddCarry [] (y:ys) ci = do
  (so, co) <- halfAdd y ci
  ss <- rippleAddCarry [] ys co
  return (so : ss)
rippleAddCarry (x:xs) [] ci = do
  (so, co) <- halfAdd x ci
  ss <- rippleAddCarry xs [] co
  return (so : ss)
rippleAddCarry (x:xs) (y:ys) ci = do
  (so, co) <- fullAdd x y ci
  ss <- rippleAddCarry xs ys co
  return (so : ss)

-- ripple carry adder
rippleAdd :: Adder
rippleAdd xs [] = return xs
rippleAdd [] ys = return ys
rippleAdd (x:xs) (y:ys) = do
  debugComment $ "Start RA: " ++ show ((x:xs), (y:ys))
  (s0, c0) <- halfAdd x y
  ss <- rippleAddCarry xs ys c0
  debugComment $ "End RA: " ++ show ((x:xs), (y:ys)) ++ " -> " ++ show (s0:ss)
  return (s0 : ss)

prop_rippleAdd :: Integral n => NonNegative n -> NonNegative n -> Bool
prop_rippleAdd (NonNegative x) (NonNegative y) =
  x + y == (fromJust . fromBits . eval $ rippleAdd (toBits x) (toBits y))

measure f = go 0
  where go i = (n-1, m) : go (i+1)
          where m = countClauses d
                (n, d) = execRWS se (cnfDebug, maxThree, minThree) 1
                se = do p <- replicateM i nextVar
                        q <- replicateM (i+1) nextVar
                        f p q

-- ripple carry subtraction
-- WARNING: computes answer in two's complement with a sign bit
rippleSub :: Subtractor
rippleSub xs ys = do
  debugComment $ "Start RS: " ++ show (xs, ys)
  let ys' = map neg ys ++ replicate (length xs - length ys) (Val True)
  zs <- rippleAddCarry xs ys' (Val True)
  let res  = init zs
      sign = last zs
  debugComment $ "End RS: " ++ show (xs, ys) ++ " -> " ++ show (res, neg sign)
  return (res, neg sign)

prop_rippleSub :: Integral n => NonNegative n -> NonNegative n -> Bool
prop_rippleSub (NonNegative x) (NonNegative y)
  | x < y     = sign == Val True
  | otherwise = sign == Val False && x - y == fromJust (fromBits res)
  where (res, sign) = eval $ rippleSub (toBits x) (toBits y)

-- lookahead carry unit
lcu :: Bit -> Bit -> SymEval ([Bit], [Bit], Bit, Bit)
lcu x y = do
  debugComment $ "Start LCU: " ++ show (x, y)
  s <- xorGate x y
  g <- andGate x y
  p <- orGate x y
  debugComment $ "End LCU: " ++ show (x, y) ++ " -> " ++ show ([s], [neg s], g, p)
  return ([s], [neg s], g, p)

-- carry-lookahead adder
cla :: Adder
cla [] [] = return []
cla xs ys = do
  debugComment $ "Start CLA: " ++ show (xs, ys)
  zs <- zipWithMBits lcu xs ys
  (s, _, g, _) <- reduce zs
  debugComment $ "End CLA: " ++ show (xs, ys) ++ " -> " ++ show (s ++ [g])
  return $ s ++ [g]
    where reduce [z] = return z
          reduce zs' = joinPairs (chunksOf 2 zs') >>= reduce
          joinPairs [] = return []
          joinPairs [[z]] = return [z]
          joinPairs ([x, y]:zs') = do
            z <- combine x y
            zs'' <- joinPairs zs'
            return (z : zs'')
          combine (s0, t0, g0, p0) (s1, t1, g1, p1) = do
            s' <- muxNGate g0 s1 t1
            t' <- muxNGate p0 s1 t1
            g <- andGate g0 p1 >>= orGate g1
            p <- andGate p0 p1 >>= orGate g1
            return (s0 ++ s', t0 ++ t', g, p)

prop_cla :: Integral n => NonNegative n -> NonNegative n -> Bool
prop_cla (NonNegative x) (NonNegative y) =
  x + y == (fromJust . fromBits . eval $ cla (toBits x) (toBits y))

type MA = [Bit] -> [Bit] -> SymEval [Bit]
type Multiplier = Adder -> MA
type Divider = Subtractor -> DS
type DS = [Bit] -> [Bit] -> SymEval ([Bit], [Bit])

-- Adds the numbers column-wise, minimizing the number of half-adders required
longMult :: MA
longMult xs ys = do
  debugComment $ "Start LM: " ++ show (xs, ys)
  zs <- mapM (\y -> mapM (andGate y) xs) ys
  r <- addCols DL.empty . map shrinkBits . transpose . zipWith shift [0..] $ zs
  debugComment $ "End LM: " ++ show (xs, ys) ++ " -> " ++ show r
  return $ DL.toList r
  where addCols acc [] = return acc
        addCols acc [[]] = return acc
        addCols acc [x] = do (z, cs) <- addCol [] x
                             addCols (DL.snoc acc z) [cs]
        addCols acc (x:y:ys) = do (z, cs) <- addCol [] x
                                  addCols (DL.snoc acc z) ((cs++y):ys)
        addCol cs [x] = return (x, cs)
        addCol cs [x,y] = do (s, c) <- halfAdd x y
                             return (s, c:cs)
        addCol cs (x:y:z:zs) = do (s, c) <- fullAdd x y z
                                  addCol (c:cs) (s:zs)

testLongMult = do
  p <- replicateM 4 nextVar
  q <- replicateM 4 nextVar
  longMult p q

--longDiv sub xs ys = quotRem xs ys
longDiv :: Subtractor -> [Bit] -> [Bit] -> SymEval ([Bit], [Bit])
longDiv sub xs ys = go [] (length xs - 2) xs
  where go acc i rem
          | i < 0     = return (acc, rem)
          | otherwise = do
              (res, sign) <- sub rem (shift i ys)
              rem1 <- zipWithM (muxGate sign) res rem
              let (rem2, z) = splitAt (length ys + i) rem1
              mapM_ assertFalseGate z
              go (neg sign : acc) (i-1) rem2

prop_longDiv :: Integral n => NonNegative n -> Positive n -> Bool
prop_longDiv (NonNegative a) (Positive b) =
  a `quotRem` b == (q, r)
  where q = fromJust (fromBits q')
        r = fromJust (fromBits r')
        (q', r') = eval $ longDiv rippleSub (toBits a) (toBits b)

-- factorMult :: Multiplier -> [Bool] -> SymEval ()
factorMult :: (Integral n, Show n) => MA -> n -> SymEval ()
factorMult mult n = do
  let out = toBits n
      len = length out
  addComment $ "factoring " ++ show n
  -- |p| <= len-1 ensures p /= out
  p <- replicateM (len-1) nextVar
  addComment $ "p: " ++ show p
  -- |q| <= ceil(len/2) ensures p > q or |p| == |q|
  q <- replicateM ((len+1) `quot` 2) nextVar
  addComment $ "q: " ++ show q
  pq <- mult p q
  zipWithMBits assertEqGate out pq
  addComment $ "pq: " ++ show pq ++ " == " ++ show out


-- factorRSA simplifies the instance somewhat by assuming the factors
-- differ at length at most one bit and are both prime
-- factorRSA :: Multiplier -> [Bool] -> SymEval ()
factorRSA :: (Integral n, Show n) => MA -> n -> SymEval ()
factorRSA mult n = do
  let out = toBits n
      outLen = length out
      pLen = (outLen + 1) `quot` 2
      qLen = (outLen + 2) `quot` 2
  addComment $ "factoring " ++ show n
  p <- if out!!0 == Val False
         then replicateM pLen nextVar
         else liftM (Val True :) (replicateM (pLen-1) nextVar)
  addComment $ "p: " ++ show p
  q <- if out!!0 == Val False && out!!1 == Val False
         then replicateM qLen nextVar
         else liftM (Val True :) (replicateM (qLen-1) nextVar)
  addComment $ "q: " ++ show q
  pq <- mult p q
  zipWithMBits assertEqGate out pq
  addComment $ "pq: " ++ show pq ++ " == " ++ show out

-- factorMultiRSA is similar to factorRSA, but it allows for multiple
-- semiprimes.  The instance is satisfied if the solver was able to
-- find the factors of any semi-prime, not just a specific one.
factorMultiRSA :: (Integral n, Show n) => MA -> [n] -> SymEval ()
factorMultiRSA mult ns = do
  let m = length ns
      outs = map toBits ns
      outLens = map length outs
      outLen = head outLens
      pLen = (outLen + 1) `quot` 2 - 1
      qLen = (outLen + 2) `quot` 2 - 1
  when (all (== outLen) (tail outLens)) $ do
    debugComment $ "creating multiplication circuit with " ++ show outLen ++ "-bit output"
    -- assuming odd prime-factors
    p <- liftM (Val True :) (replicateM qLen nextVar)
    q <- liftM (Val True :) (replicateM pLen nextVar)
    addComment $ "p: " ++ show p
    addComment $ "q: " ++ show q
    pq <- mult p q
    ys <- zipWithM eqNGate (replicate m pq) outs
    assertOrNGate ys

-- factorDivP :: Divider -> [Bool] -> SymEval ()
-- divide by p: the (n-1) bit number
--
-- TODO: minor optimization: this computes q from p, while we're only
-- interested in the remainder
factorDivP :: (Integral n, Show n) => DS -> n -> SymEval ()
factorDivP div n = do
  let pq = toBits n
      len = length pq
  addComment $ "factoring " ++ show n ++ "; in " ++ show pq
  p <- replicateM (len-1) nextVar
  addComment $ "p: " ++ show p
  (q, r) <- div pq p
  addComment $ "q: " ++ show q
  mapM_ assertFalseGate r
  addComment $ "remainder (zero): " ++ show r

-- factorDivQ :: Divider -> [Bool] -> SymEval ()
-- divide by Q: the (n-1) bit number
factorDivQ :: (Integral n, Show n) => DS -> n -> SymEval ()
factorDivQ div n = do
  let pq = toBits n
      len = length pq
  addComment $ "factoring " ++ show n ++ "; in " ++ show pq
  q <- replicateM ((len+1) `quot` 2) nextVar
  addComment $ "q: " ++ show q
  (p, r) <- div pq q
  addComment $ "p: " ++ show p
  mapM_ assertFalseGate r
  addComment $ "remainder (zero): " ++ show r

main = do
  args <- getArgs
  pname <- getProgName
  case args of
    []  -> error $ "Usage " ++ pname ++ " [N]+"
    [x] -> putStr . seToDimacs . factorRSA longMult . read $ x
    xs  -> putStr . seToDimacs . factorMultiRSA longMult . map read $ xs

return []
runTests = $quickCheckAll
