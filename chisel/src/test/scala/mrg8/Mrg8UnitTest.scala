package mrg8

import java.io.File

import chisel3.iotesters
import chisel3.iotesters.{ChiselFlatSpec, Driver, PeekPokeTester}

class Mrg8UnitTester(c: Mrg8) extends PeekPokeTester(c) {
  /**
    * compute the gcd and the number of steps it should take to do it
    *
    * @param a positive integer
    * @param b positive integer
    * @return the GCD of a and b
    */
  def computeMrg8(a: Int, b: Int): (Int, Int) = {
    var x = a
    var y = b
    var depth = 1
    while(y > 0 ) {
      if (x > y) {
        x -= y
      }
      else {
        y -= x
      }
      depth += 1
    }
    (x, depth)
  }

  private val mrg8 = c

  for(i <- 1 to 40 by 3) {
    for (j <- 1 to 40 by 7) {
      poke(mrg8.io.value1, i)
      poke(mrg8.io.value2, j)
      poke(mrg8.io.loadingValues, 1)
      step(1)
      poke(mrg8.io.loadingValues, 0)

      val (expected_mrg8, steps) = computeMrg8(i, j)

      step(steps - 1) // -1 is because we step(1) already to toggle the enable
      expect(mrg8.io.out, expected_mrg8)
      expect(mrg8.io.outputValid, 1)
    }
  }
}

/**
  * This is a trivial example of how to run this Specification
  * From within sbt use:
  * {{{
  * testOnly mrg8.Mrg8Tester
  * }}}
  * From a terminal shell use:
  * {{{
  * sbt 'testOnly mrg8.Mrg8Tester'
  * }}}
  */
class Mrg8Tester extends ChiselFlatSpec {
  // Disable this until we fix isCommandAvailable to swallow stderr along with stdout
  private val backendNames = if(firrtl.FileUtils.isCommandAvailable(Seq("verilator", "--version"))) {
    Array("firrtl", "verilator")
  }
  else {
    Array("firrtl")
  }
  for ( backendName <- backendNames ) {
    "GCD" should s"calculate proper greatest common denominator (with $backendName)" in {
      Driver(() => new Mrg8, backendName) {
        c => new Mrg8UnitTester(c)
      } should be (true)
    }
  }

  "Basic test using Driver.execute" should "be used as an alternative way to run specification" in {
    iotesters.Driver.execute(Array(), () => new Mrg8) {
      c => new Mrg8UnitTester(c)
    } should be (true)
  }

  if(backendNames.contains("verilator")) {
    "using --backend-name verilator" should "be an alternative way to run using verilator" in {
      iotesters.Driver.execute(Array("--backend-name", "verilator"), () => new Mrg8) {
        c => new Mrg8UnitTester(c)
      } should be(true)
    }
  }

  "running with --is-verbose" should "show more about what's going on in your tester" in {
    iotesters.Driver.execute(Array("--is-verbose"), () => new Mrg8) {
      c => new Mrg8UnitTester(c)
    } should be(true)
  }

  /**
    * By default verilator backend produces vcd file, and firrtl and treadle backends do not.
    * Following examples show you how to turn on vcd for firrtl and treadle and how to turn it off for verilator
    */

  "running with --generate-vcd-output on" should "create a vcd file from your test" in {
    iotesters.Driver.execute(
      Array("--generate-vcd-output", "on", "--target-dir", "test_run_dir/make_a_vcd", "--top-name", "make_a_vcd"),
      () => new Mrg8
    ) {

      c => new Mrg8UnitTester(c)
    } should be(true)

    new File("test_run_dir/make_a_vcd/make_a_vcd.vcd").exists should be (true)
  }

  "running with --generate-vcd-output off" should "not create a vcd file from your test" in {
    iotesters.Driver.execute(
      Array("--generate-vcd-output", "off", "--target-dir", "test_run_dir/make_no_vcd", "--top-name", "make_no_vcd",
      "--backend-name", "verilator"),
      () => new Mrg8
    ) {

      c => new Mrg8UnitTester(c)
    } should be(true)

    new File("test_run_dir/make_no_vcd/make_a_vcd.vcd").exists should be (false)

  }

}
