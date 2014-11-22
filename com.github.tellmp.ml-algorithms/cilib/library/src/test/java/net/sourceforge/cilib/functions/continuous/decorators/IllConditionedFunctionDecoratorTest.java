/**           __  __
 *    _____ _/ /_/ /_    Computational Intelligence Library (CIlib)
 *   / ___/ / / / __ \   (c) CIRG @ UP
 *  / /__/ / / / /_/ /   http://cilib.net
 *  \___/_/_/_/_.___/
 */
package net.sourceforge.cilib.functions.continuous.decorators;

import net.sourceforge.cilib.controlparameter.ConstantControlParameter;
import net.sourceforge.cilib.functions.continuous.unconstrained.Spherical;
import net.sourceforge.cilib.type.types.container.Vector;
import org.junit.Test;
import static org.junit.Assert.*;

public class IllConditionedFunctionDecoratorTest {

    @Test
    public void testApply() {
        Spherical s = new Spherical();
        IllConditionedFunctionDecorator ic = new IllConditionedFunctionDecorator();
        ic.setAlpha(ConstantControlParameter.of(1.0));
        ic.setFunction(s);

        Vector v = Vector.of(1.0, 2.0);
        assertEquals(5.0, ic.apply(v), 0.0);

        ic.setAlpha(ConstantControlParameter.of(2.0));
        assertEquals(9.0, ic.apply(v), 0.0000001);

        ic.setAlpha(ConstantControlParameter.of(3.0));
        assertEquals(13.0, ic.apply(v), 0.0000001);
    }
}
