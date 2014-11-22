/**           __  __
 *    _____ _/ /_/ /_    Computational Intelligence Library (CIlib)
 *   / ___/ / / / __ \   (c) CIRG @ UP
 *  / /__/ / / / /_/ /   http://cilib.net
 *  \___/_/_/_/_.___/
 */
package net.sourceforge.cilib.pso.velocityprovider;

import net.sourceforge.cilib.algorithm.AbstractAlgorithm;
import net.sourceforge.cilib.controlparameter.ConstantControlParameter;
import net.sourceforge.cilib.controlparameter.ControlParameter;
import net.sourceforge.cilib.entity.Topologies;
import net.sourceforge.cilib.entity.comparator.SocialBestFitnessComparator;
import net.sourceforge.cilib.math.random.generator.Rand;
import net.sourceforge.cilib.problem.solution.Fitness;
import net.sourceforge.cilib.problem.solution.InferiorFitness;
import net.sourceforge.cilib.pso.PSO;
import net.sourceforge.cilib.pso.particle.Particle;
import net.sourceforge.cilib.type.types.Bounds;
import net.sourceforge.cilib.type.types.container.Vector;

/**
 * An implementation of the Guaranteed Convergence PSO algorithm. The GCPSO is a simple extension
 * to the normal PSO algorithm and the modifications to the algorithm is implemented as
 * a simple {@link VelocityProvider}.
 * <p>
 * References:
 * <p><ul><li>
 * F. van den Bergh and A. Engelbrecht, "A new locally convergent particle swarm optimizer,"
 * in Proceedings of IEEE Conference on Systems, Man and Cybernetics,
 * (Hammamet, Tunisia), Oct. 2002.
 * </li><li>
 * F. van den Bergh, "An Analysis of Particle Swarm Optimizers,"
 * PhD thesis, Department of Computer Science,
 * University of Pretoria, South Africa, 2002.
 * </li></ul>
 * <p>
 * TODO: The Rho value should be a vector to hold the rho value for each dimension!
 * <p>
 * It is very important to realise the importance of the <code>rho</code> values. <code>rho</code>
 * determines the local search size of the global best particle and depending on the domain
 * this could result in poor performance if the <code>rho</code> value is too small or too large depending
 * on the specified problem domain. For example, a <code>rho</code> value of 1.0 is not a good
 * value within problems which have a domain that spans <code>[0,1]</code>
 */
public class GCVelocityProvider implements VelocityProvider {

    private static final long serialVersionUID = 5985694749940610522L;

    private VelocityProvider delegate;

    private ControlParameter inertiaWeight;
    private ControlParameter rhoLowerBound;
    private ControlParameter rho;

    private int successCount;
    private int failureCount;
    private int successCountThreshold;
    private int failureCountThreshold;

    private Fitness oldFitness;
    private ControlParameter rhoExpandCoefficient;
    private ControlParameter rhoContractCoefficient;

    /**
     * Create an instance of the GC Velocity Update strategy.
     */
    public GCVelocityProvider() {
        this.delegate = new StandardVelocityProvider();

        this.inertiaWeight = ConstantControlParameter.of(0.729844);

        this.rho = ConstantControlParameter.of(1.0);
        this.rhoLowerBound = ConstantControlParameter.of(1.0e-323);

        this.successCount = 0;
        this.failureCount = 0;
        this.successCountThreshold = 15;
        this.failureCountThreshold = 5;

        this.oldFitness = InferiorFitness.instance();
        this.rhoExpandCoefficient = ConstantControlParameter.of(1.2);
        this.rhoContractCoefficient = ConstantControlParameter.of(0.5);
    }

    /**
     * Copy constructor. Copy the given instance.
     * @param copy The instance to copy.
     */
    public GCVelocityProvider(GCVelocityProvider copy) {
        this.delegate = copy.delegate.getClone();
        this.inertiaWeight = copy.inertiaWeight.getClone();

        this.rho = copy.rho.getClone();
        this.rhoLowerBound = copy.rhoLowerBound.getClone();

        this.successCount = copy.successCount;
        this.failureCount = copy.failureCount;
        this.successCountThreshold = copy.successCountThreshold;
        this.failureCountThreshold = copy.failureCountThreshold;

        this.oldFitness = copy.oldFitness.getClone();
        this.rhoExpandCoefficient = copy.rhoExpandCoefficient.getClone();
        this.rhoContractCoefficient = copy.rhoContractCoefficient.getClone();
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public GCVelocityProvider getClone() {
        return new GCVelocityProvider(this);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Vector get(Particle particle) {
        PSO pso = (PSO) AbstractAlgorithm.get();
        final Particle globalBest = Topologies.getBestEntity(pso.getTopology(), new SocialBestFitnessComparator<Particle>());
        Vector result;

        if (particle == globalBest) {
            final Vector velocity = (Vector) particle.getVelocity();
            final Vector position = (Vector) particle.getPosition();
            final Vector globalGuide = (Vector) particle.getGlobalGuide();

            Vector.Builder builder = Vector.newBuilder();
            for (int i = 0; i < velocity.size(); ++i) {
                double component = -position.doubleValueOf(i) + globalGuide.doubleValueOf(i)
                        + this.inertiaWeight.getParameter() * velocity.doubleValueOf(i)
                        + this.rho.getParameter() * (1 - 2 * Rand.nextDouble());
                builder.add(component);
            }

            this.oldFitness = particle.getFitness().getClone(); // Keep a copy of the old Fitness object - particle.calculateFitness() within the IterationStrategy resets the fitness value

            result = builder.build();
        }
        else {
            result =  this.delegate.get(particle);
        }

        updateControlParameters(particle);
        return result;
    }

    /**
     * Updates certain control parameters for this velocity provider.
     * @param particle
     */
    public void updateControlParameters(Particle particle) {
        // Remember NOT to reset the rho value to 1.0
        PSO pso = (PSO) AbstractAlgorithm.get();

        if (particle == Topologies.getBestEntity(pso.getTopology(), new SocialBestFitnessComparator<Particle>())) {
            Fitness newFitness = particle.getFitnessCalculator().getFitness(particle);

            if (!newFitness.equals(oldFitness)) {
                this.failureCount = 0;
                this.successCount++;
            } else {
                this.successCount = 0;
                this.failureCount++;
            }

            updateRho((Vector) particle.getPosition());
            return;
        }

        this.failureCount = 0;
        this.successCount = 0;
    }

    /**
     * Update the <code>rho</code> value.
     * @param position
     */
    private void updateRho(Vector position) { // the Rho value is problem and dimension dependent
        double tmp = 0.0;

        Bounds component = position.boundsOf(0);
        double average = (component.getUpperBound() - component.getLowerBound()) / this.rhoExpandCoefficient.getParameter();

        if (this.successCount >= this.successCountThreshold) {
            tmp = this.rhoExpandCoefficient.getParameter() * this.rho.getParameter();
        }
        if (this.failureCount >= this.failureCountThreshold) {
            tmp = this.rhoContractCoefficient.getParameter() * this.rho.getParameter();
        }

        if (tmp <= this.rhoLowerBound.getParameter()) {
            tmp = this.rhoLowerBound.getParameter();
        }
        if (tmp >= average) {
            tmp = average;
        }

        this.rho = ConstantControlParameter.of(tmp);
    }

    public VelocityProvider getDelegate() {
        return this.delegate;
    }

    public void setDelegate(VelocityProvider delegate) {
        this.delegate = delegate;
    }

    /**
     * Get the lower-bound value for <code>rho</code>.
     * @return The lower-bound value for <code>rho</code>.
     */
    public ControlParameter getRhoLowerBound() {
        return this.rhoLowerBound;
    }

    /**
     * Set the lower-bound value for <code>rho</code>.
     * @param rhoLowerBound The lower-bound to set.
     */
    public void setRhoLowerBound(ControlParameter rhoLowerBound) {
        this.rhoLowerBound = rhoLowerBound;
    }

    /**
     * Get the current value for <code>rho</code>.
     * @return The current value for <code>rho</code>.
     */
    public ControlParameter getRho() {
        return this.rho;
    }

    /**
     * Set the value for <code>rho</code>.
     * @param rho The value to set.
     */
    public void setRho(ControlParameter rho) {
        this.rho = rho;
    }

    /**
     * Get the count of success threshold.
     * @return The success threshold.
     */
    public int getSuccessCountThreshold() {
        return this.successCountThreshold;
    }

    /**
     * Set the threshold of success count value.
     * @param successCountThreshold The value to set.
     */
    public void setSuccessCountThreshold(int successCountThreshold) {
        this.successCountThreshold = successCountThreshold;
    }

    /**
     * Get the count of failure threshold.
     * @return The failure threshold.
     */
    public int getFailureCountThreshold() {
        return this.failureCountThreshold;
    }

    /**
     * Set the count of failure threshold.
     * @param failureCountThreshold The value to set.
     */
    public void setFailureCountThreshold(int failureCountThreshold) {
        this.failureCountThreshold = failureCountThreshold;
    }

    /**
     * Get the coefficient value for <code>rho</code> expansion.
     * @return The expansion coefficient value.
     */
    public ControlParameter getRhoExpandCoefficient() {
        return this.rhoExpandCoefficient;
    }

    /**
     * Set the value of the coefficient of expansion.
     * @param rhoExpandCoefficient The value to set.
     */
    public void setRhoExpandCoefficient(ControlParameter rhoExpandCoefficient) {
        this.rhoExpandCoefficient = rhoExpandCoefficient;
    }

    /**
     * Get the coefficient value for <code>rho</code> contraction.
     * @return The contraction coefficient value.
     */
    public ControlParameter getRhoContractCoefficient() {
        return this.rhoContractCoefficient;
    }

    /**
     * Set the contraction coefficient value.
     * @param rhoContractCoefficient The value to set.
     */
    public void setRhoContractCoefficient(ControlParameter rhoContractCoefficient) {
        this.rhoContractCoefficient = rhoContractCoefficient;
    }

    /**
     * Set the inertia weight value.
     * @param inertiaWeight The value to set.
     */
    public void setInertiaWeight(ControlParameter inertiaWeight) {
        this.inertiaWeight = inertiaWeight;
    }
}
