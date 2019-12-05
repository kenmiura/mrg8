package mrg8

import chisel3._

/**
  * MRG8 random number generator
  * This version is the unpipelined version

  *  Needs a method to preload the Xn registers to init
  *  Second method is to set a jumpahead distance and jump
  */
class Mrg8 extends Module {
  val io = IO(new Bundle {
    val value1        = Input(UInt(32.W))
    val value2        = Input(UInt(32.W))
    val loadingValues = Input(Bool())
    val out    = Output(UInt(32.W))
    val outputValid   = Output(Bool())
  })

  val x  = Reg(UInt())
  val y  = Reg(UInt())

  private val x0 = RegNext(io.out)
  private val x1 = RegNext(x0)
  private val x2 = RegNext(x1)
  private val x3 = RegNext(x2)
  private val x4 = RegNext(x3)
  private val x5 = RegNext(x4)
  private val x6 = RegNext(x5)
  private val x7 = RegNext(x6)
/* assign now to inputs for mult operation */

  private val a0 = RegInit(1089656042.U(32.W))
  private val a1 = RegInit(1906537547.U(32.W))
  private val a2 = RegInit(1764115693.U(32.W))
  private val a3 = RegInit(1304127872.U(32.W))
  private val a4 = RegInit(189748160.U(32.W))
  private val a5 = RegInit(1984088114.U(32.W))
  private val a6 = RegInit(626062218.U(32.W))
  private val a7 = RegInit(1927846343.U(32.W))

  when(x > y) { x := x - y }
    .otherwise { y := y - x }

  when(io.loadingValues) {
    x := io.value1
    y := io.value2
  }

  io.out := x
  io.outputValid := y === 0.U
}
