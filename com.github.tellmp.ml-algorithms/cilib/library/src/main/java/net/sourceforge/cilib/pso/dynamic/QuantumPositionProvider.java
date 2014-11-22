/**           __  __
 *    _____ _/ /_/ /_    Computational Intelligence Library (CIlib)
 *   / ___/ / / / __ \   (c) CIRG @ UP
 *  / /__/ / / / /_/ /   http://cilib.net
 *  \___/_/_/_/_.___/
 */
package net.sourceforge.cilib.pso.dynamic;

import java.util.Arrays;
import net.sourceforge.cilib.algorithm.AbstractAlgorithm;
import net.sourceforge.cilib.controlparameter.ConstantControlParameter;
import net.sourceforge.cilib.controlparameter.ControlParameter;
import net.sourceforge.cilib.math.random.ProbabilityDistributionFunction;
import net.sourceforge.cilib.math.random.UniformDistribution;
import net.sourceforge.cilib.pso.particle.Particle;
import net.sourceforge.cilib.pso.positionprovider.PositionProvider;
import net.sourceforge.cilib.pso.positionprovider.StandardPositionProvider;
import net.sourceforge.cilib.type.types.container.Vector;

/**
 * Position provider for QSO (Quantum PSO). Implemented according to paper by
 * Blackwell and Branke, "Multiswarms, Exclusion, and Anti-Convergence in
 * Dynamic Environments."
 *
 */
public class QuantumPositionProvider implements PositionProvider {

    private static final long serialVersionUID = -7844226788317206737L;

    private static final double EPSILON = 0.000000001;

    private ControlParameter radius;
    private ProbabilityDistributionFunction randomiser;
    private Vector nucleus;

    private PositionProvider delegate;

    public QuantumPositionProvider() {
        this.radius = ConstantControlParameter.of(5);
        this.randomiser = new UniformDistribution();
        this.delegate = new StandardPositionProvider();
    }

    public QuantumPositionProvider(QuantumPositionProvider copy) {
        this.radius = copy.radius;
        this.randomiser = copy.randomiser;
        this.delegate = copy.delegate.getClone();
    }

    @Override
    public QuantumPositionProvider getClone() {
        return new QuantumPositionProvider(this);
    }

    /**
     * Update particle position; do it in a standard way if the particle is neutral, and
     * in a quantum way if the particle is charged. The "quantum" way entails sampling the
     * position from a uniform distribution : a spherical cloud around gbest with a radius r.
     * @param particle the particle to update position of
     */
    @Override
    public Vector get(Particle particle) {
        ChargedParticle checkChargeParticle = (ChargedParticle) particle;
        if (checkChargeParticle.getCharge() < EPSILON) { // the particle is neutral
            return this.delegate.get(particle);
        } else { // the particle is charged
            //based on the Pythagorean theorem,
            //the following code breaks the square of the radius distance into smaller
            //parts that are then "distributed" among the dimensions of the problem.
            //the position of the particle is determined in each dimension by a random number
            //between 0 and the part of the radius assigned to that dimension
            //This ensures that the quantum particles are placed randomly within the
            //multidimensional sphere determined by the quantum radius.

            this.nucleus = (Vector) AbstractAlgorithm.get().getBestSolution().getPosition();

            double distance = Math.pow(this.radius.getParameter(), 2); //square of the radius
            int dimensions = particle.getDimension();
            double[] pieces = new double[dimensions]; // break up of the distance
            pieces[dimensions - 1] = distance;
            for (int i = 0; i < dimensions - 1; i++) {
                pieces[i] = this.randomiser.getRandomNumber(0, distance);
            }//for
            Arrays.sort(pieces);
            int sign = 1;
            if (this.randomiser.getRandomNumber() <= 0.5) {
                sign = -1;
            }//if
            //deals with first dimension
            Vector.Builder builder = Vector.newBuilder();
            builder.add(this.nucleus.doubleValueOf(0) + sign * this.randomiser.getRandomNumber(0, Math.sqrt(pieces[0])));
            //deals with the other dimensions
            for (int i = 1; i < dimensions; i++) {
                sign = 1;
                if (this.randomiser.getRandomNumber() <= 0.5) {
                    sign = -1;
                }//if
                double rad = Math.sqrt(pieces[i] - pieces[i - 1]);
                double dis = this.randomiser.getRandomNumber(0, rad);
                double newpos = this.nucleus.doubleValueOf(i) + sign * dis;
                builder.add(newpos);
            }//for
            return builder.build();
        }//else
    }

    public void setDelegate(PositionProvider delegate) {
        this.delegate = delegate;
    }

    public PositionProvider getDelegate() {
        return this.delegate;
    }

    public Vector getNucleus() {
        return this.nucleus;
    }

    public void setNucleus(Vector nucleus) {
        this.nucleus = nucleus;
    }

    /**
     * @return the radius
     */
    public ControlParameter getRadius() {
        return this.radius;
    }

    /**
     * @param radius the radius to set
     */
    public void setRadius(ControlParameter radius) {
        //Preconditions.checkArgument(radius.getParameter() >= 0, "Radius of the electron cloud can not be negative");
        this.radius = radius;
    }
}
