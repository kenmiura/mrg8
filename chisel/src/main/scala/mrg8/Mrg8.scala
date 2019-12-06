package mrg8

import chisel3._

/**
  * MRG8 random number generator
  * This version is the unpipelined version
  * Might be hard to meet timing

  *  Needs a method to preload the Xn registers to init
  *  Second method is to set a jumpahead distance and jump
  */
class Mrg8 extends Module {
  val io = IO(new Bundle { /* many of these are holdovers from GCD demo.  only "loadingValues" "out" and "nextNumber" are used by MRG8 */
    val value1        = Input(UInt(32.W))
    val value2        = Input(UInt(32.W))
    val loadingValues = Input(Bool())
    val out    = Output(UInt(32.W))
    val outputValid   = Output(Bool())
    val nextNumber    = Input(Bool())
  })

  val x  = Reg(UInt())
  val y  = Reg(UInt())

/* Random Numbers */
  private val x0 = Reg(UInt(31.W))
  private val x1 = Reg(UInt(31.W))
  private val x2 = Reg(UInt(31.W))
  private val x3 = Reg(UInt(31.W))
  private val x4 = Reg(UInt(31.W))
  private val x5 = Reg(UInt(31.W))
  private val x6 = Reg(UInt(31.W))
  private val x7 = Reg(UInt(31.W))
/* assign now to inputs for mult operation */

/* Constants matrix (currently vector) */
/* Pre-initialized with matrix constants */
  private val a0 = RegInit(1089656042.U(31.W))
  private val a1 = RegInit(1906537547.U(31.W))
  private val a2 = RegInit(1764115693.U(31.W))
  private val a3 = RegInit(1304127872.U(31.W))
  private val a4 = RegInit(189748160.U(31.W))
  private val a5 = RegInit(1984088114.U(31.W))
  private val a6 = RegInit(626062218.U(31.W))
  private val a7 = RegInit(1927846343.U(32.W))
  
  /* Multiply by Constants a0-a7 */
  private val xa0 = Reg(UInt(62.W))
  private val xa1 = Reg(UInt(62.W))
  private val xa2 = Reg(UInt(62.W))
  private val xa3 = Reg(UInt(62.W))
  private val xa4 = Reg(UInt(62.W))
  private val xa5 = Reg(UInt(62.W))
  private val xa6 = Reg(UInt(62.W))
  private val xa7 = Reg(UInt(62.W))
  
  /* First stage reduction */
  private val ra01 = Reg(UInt(63.W))
  private val ra23 = Reg(UInt(63.W))
  private val ra45 = Reg(UInt(63.W))
  private val ra67 = Reg(UInt(63.W))
  
  /* Second Stage Reduction */
  private val rb0123 = Reg(UInt(64.W))
  private val rb4567 = Reg(UInt(64.W))
  
  /* Last Stage Reduction */
  private val rc = Reg(UInt(65.W))
  
  /* first stage Normalization */
  private val n1 = Reg(UInt(34.W))
  /* second stage Normalization */
  private val n2 = Reg(UInt(32.W))
  /* third stage Normalization */
  private val n3 = Reg(UInt(31.W))
  
  
  when(io.nextNumber){ /* enable pulse to compute next random number */
    /* this implements all stages as combinational logic.
       that means it will be next to impossible to meet timing.
       Should I manually/explicitly pipeline, or should it be handled during synthesis? */
    /* Multiply the RNG regs by the coefficient array */
    xa0 := x0 * a0
    xa1 := x1 * a1
    xa2 := x2 * a2
    xa3 := x3 * a3
    xa4 := x4 * a4
    xa5 := x5 * a5
    xa6 := x6 * a6
    xa7 := x7 * a7
    
    /* First (a) stage reduction */
    ra01 := xa0 + xa1
    ra23 := xa2 + xa3
    ra45 := xa4 + xa5
    ra67 := xa6 + xa7
    
    /* Second (b) stage reduction */
    rb0123 := ra01 + ra23
    rb4567 := ra45 + ra67
    
    /* Final (c) stage reduction */
    rc := rb0123 + rb4567
    
    /* Now for the Normalizztions */
    n1 := rc(64,33) + rc(32,0) /* upper 32 bits + lower 33 bits for output of 34 bits */
    n2 := n1(33,31) + n1(30,0) /* upper 3 bits + lower 31 bits for output to 32 bits */
    n3 := n2(31,31) + n2(30,0) /* final normalization before output of new random number */
    
    io.out := n3 /* redundant, but copying from private to non-private register */
    
    /* then shift the input seeds for next iteration */
    x0 := io.out
    x1 := x0
    x2 := x1
    x3 := x2
    x4 := x3
    x5 := x4
    x6 := x5
    x7 := x6
  }

/* from original GCD demo */
  when(x > y) { x := x - y }
    .otherwise { y := y - x }

  when(io.loadingValues) {
    /* this is the old GCD code, so we can continue to fake out the tester */
    x := io.value1
    y := io.value2
    
    /* MRG8: lets also hijack this init hook to load up the inital x0-x7 values */
    /* this is a very stupid way to do it, but its a start */
    x0 := a0
    x1 := a1
    x2 := a2
    x3 := a3
    x4 := a4
    x5 := a5
    x6 := a6
    x7 := a7
  }

  io.out := x
  io.outputValid := y === 0.U
}
