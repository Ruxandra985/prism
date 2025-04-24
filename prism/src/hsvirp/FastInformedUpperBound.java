package hsvirp;

import java.util.Arrays;
import java.util.BitSet;
import java.util.HashMap;
import java.util.Map;

import explicit.POMDP;
import explicit.rewards.MDPRewards;

public class FastInformedUpperBound {
    private int maxIter = Integer.MAX_VALUE;
    private double maxTime = Integer.MAX_VALUE;
    private double belRes = Double.MIN_VALUE;
    private double initValue = 0.0;
    private HashMap<Object, Double> residuals;
    HashMap<Object, Double[]> alphaVectors = new HashMap<>();
  
    // Constructor
    public FastInformedUpperBound() {
      this.residuals = new HashMap<>();
    }
    
    public static Double computeBeliefResiduals(Double[] alpha1, Double[] alpha2) {
      Double maxRes = 0.0;
      for (int i = 0; i < alpha1.length; i++) {
          Double res = Math.abs(alpha1[i] - alpha2[i]);
          if (res > maxRes) {
              maxRes = res;
          }
      }
      return maxRes;
    }
    public void update(POMDP<Double> pomdp, MDPRewards<Double> mdpRewards, BitSet target, BitSet remain) {
      Double[] alphaTemporary = new Double[pomdp.getNumStates()];
      Arrays.fill(alphaTemporary, 0.0);
      
      for (Object actionName: alphaVectors.keySet()) {
        
        Arrays.fill(alphaTemporary, 0.0);
        
        for (int state = 0 ; state < pomdp.getNumStates() ; state++) {
          
          int action = pomdp.getChoiceByAction(state, actionName);
          
          if (action == -1 || (remain != null && !remain.get(state)))
            continue; // action not possible from state
          
          
          Double reward = mdpRewards.getTransitionReward(state, action);
          
          assert (!Double.isInfinite(reward));
          
          if (target.get(state)) {
            alphaTemporary[state] = 0.0; // this should already be 0...??
          }
          else {
            double[] certainStateBelief = new double[pomdp.getNumStates()];
            certainStateBelief[state] = 1.0;
            HashMap<Integer, Double> observationProbs = pomdp.computeObservationProbsAfterAction(certainStateBelief, action);
            
            Double tmp = 0.0;
            
            for (Map.Entry<Integer, Double> entry : observationProbs.entrySet()) {
              Integer observation = entry.getKey();
              Double obsProb = entry.getValue();
              
              Double vMax = Double.NEGATIVE_INFINITY;
              
              for (Map.Entry<Object, Double[]> alphaVecIt : alphaVectors.entrySet()) {
                Object alphaVAction = alphaVecIt.getKey();
                Double[] alphaVec = alphaVecIt.getValue();
                
                Double vCurr = 0.0;
                double[] successor = pomdp.getBeliefInDistAfterChoice(certainStateBelief, action);
              
                for (int succState = 0 ; succState < pomdp.getNumStates() ; succState++) {
                  
                  if (successor[succState] == 0.0)
                    continue;
                  
                  vCurr += obsProb * successor[succState] * alphaVec[succState];
                  
                }
                
                if (vMax < vCurr)
                  vMax = vCurr;
              
              }
              
              tmp += vMax;
            }
            alphaTemporary[state] = reward + tmp;
            
          }
        }
        residuals.put(actionName,
            computeBeliefResiduals(alphaVectors.get(actionName), alphaTemporary));
        
        alphaVectors.put(actionName, alphaTemporary.clone());
      }
      
    }

    public HashMap<Object, Double[]> computePolicy (POMDP<Double> pomdp, MDPRewards<Double> mdpRewards, BitSet target, BitSet remain) {
        long t0 = System.currentTimeMillis();

        residuals = new HashMap<>();
        
        if (Double.isFinite(initValue)) {
          for (int state = 0 ; state < pomdp.getNumStates() ; state++) {
            for (int action = 0 ; action < pomdp.getNumChoices(state); action++) {
              Object actionName = pomdp.getAction(state, action);
              residuals.put(actionName, 0.0);
              if (!alphaVectors.containsKey(actionName)) {
                alphaVectors.put(actionName, new Double[pomdp.getNumStates()]);
                Arrays.fill(alphaVectors.get(actionName), initValue);
              }
            }
          }
        }

        
        int iter = 0;
        while (iter < maxIter && (System.currentTimeMillis() - t0) / 1000.0 < maxTime) {
          update(pomdp, mdpRewards, target, remain);
          iter++;
          
          boolean smallerThanBelRes = true;
          
          for (Double residual: residuals.values()) {
            smallerThanBelRes = smallerThanBelRes && (residual < belRes);
          }
          
          if (smallerThanBelRes) 
              break;
        }
        
        return alphaVectors;
    }
    
}
