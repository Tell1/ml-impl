/**           __  __
 *    _____ _/ /_/ /_    Computational Intelligence Library (CIlib)
 *   / ___/ / / / __ \   (c) CIRG @ UP
 *  / /__/ / / / /_/ /   http://cilib.net
 *  \___/_/_/_/_.___/
 */
package net.sourceforge.cilib.functions.continuous.dynamic.moo.hef8;

import net.sourceforge.cilib.algorithm.AbstractAlgorithm;
import net.sourceforge.cilib.functions.ContinuousFunction;
import net.sourceforge.cilib.problem.FunctionOptimisationProblem;
import net.sourceforge.cilib.type.types.container.Vector;

/**
 * This function is the f2 function of the adapted F8 problem defined in the
 * following paper: H. Li and Q. Zhang. Multiobjective optimization problems
 * with complicated Pareto sets, MOEA/D and NSGA-II, IEEE Transactions on
 * Evolutionary Computation, 13(2):284-302, 2009.
 *
 * The problem has been adapted by Helbig and Engelbrecht to make it a DMOOP.
 *
 */
public class HEF8_f2 implements ContinuousFunction {

    private static final long serialVersionUID = 6369118486095689078L;
    //member
    ContinuousFunction hef8_g;
    ContinuousFunction hef8_h;
    FunctionOptimisationProblem hef8_g_problem;
    FunctionOptimisationProblem hef8_h_problem;

    //Domain("R(-1, 1)^20")

    /**
     * Sets the g function with a specified problem.
     * @param problem FunctionOptimisationProblem used for the g function.
     */
    public void setHEF8_g(FunctionOptimisationProblem problem) {
        this.hef8_g_problem = problem;
        this.hef8_g = (ContinuousFunction) problem.getFunction();
    }

    /**
     * Returns the problem used to set the g function.
     * @return hef8_g_problem FunctionOptimisationProblem used for the g
     * function.
     */
    public FunctionOptimisationProblem getHEF8_g_problem() {
        return this.hef8_g_problem;
    }

    /**
     * Sets the g function that is used in the HEF8 problem without specifying
     * the problem.
     * @param hef8_g ContinuousFunction used for the g function.
     */
    public void setHEF8_g(ContinuousFunction hef8_g) {
        this.hef8_g = hef8_g;
    }

    /**
     * Returns the g function that is used in the HEF8 problem.
     * @return hef8_g ContinuousFunction used for the g function.
     */
    public ContinuousFunction getHEF8_g() {
        return this.hef8_g;
    }

    /**
     * Sets the h function with a specified problem.
     * @param problem FunctionOptimisationProblem used for the h function.
     */
    public void setHEF8_h(FunctionOptimisationProblem problem) {
        this.hef8_h_problem = problem;
        this.hef8_h = (ContinuousFunction) problem.getFunction();
    }

    /**
     * Returns the problem used to set the h function.
     * @return hef8_h_problem FunctionOptimisationProblem used for the h
     * function.
     */
    public FunctionOptimisationProblem getHEF8_h_problem() {
        return this.hef8_h_problem;
    }

    /**
     * Sets the h function that is used in the HEF8 problem without specifying
     * the problem.
     * @param hef8_h ContinuousFunction used for the h function.
     */
    public void setHEF8_h(ContinuousFunction hef8_h) {
        this.hef8_h = hef8_h;
    }

    /**
     * Sets the f1 function that is used in the HEF8 problem without specifying
     * the problem.
     * @return hef8_h ContinuousFunction used for the h function.
     */
    public ContinuousFunction getHEF8_h() {
        return this.hef8_h;
    }

    /**
     * Evaluates the function. g*h
     */
    @Override
    public Double apply(Vector input) {
        int iteration = AbstractAlgorithm.get().getIterations();
        return apply(iteration, input);
    }

    /**
     * Evaluates the function for a specific iteration. g*h
     */
    public Double apply(int iteration, Vector input) {
        double g = ((HEF8_g) this.hef8_g).apply(input);
        double h = ((HEF8_h) this.hef8_h).apply(iteration, input);

        double value = g * h;

        return value;
    }
}
