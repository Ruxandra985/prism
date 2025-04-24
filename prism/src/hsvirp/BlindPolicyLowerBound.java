package hsvirp;

import java.util.Arrays;
import java.util.HashMap;
import explicit.POMDP;
import explicit.rewards.MDPRewards;

public class BlindPolicyLowerBound {
    private int maxIterations;
    private double maxTime;
    private double belRes;
    private HashMap<Object, Double> residuals;

    HashMap<Object, Double[]> alphaVectors = new HashMap<>();
    
    // Note: discount factor is taken as 1
    
    // Constructor
    public BlindPolicyLowerBound(int maxIter, double maxTime, double beliefResidual) {
        this.maxIterations = maxIter;
        this.maxTime = maxTime;
        this.belRes = beliefResidual;
        this.residuals = new HashMap<>();
    }

    // Default Constructor
    public BlindPolicyLowerBound() {
        this(Integer.MAX_VALUE, 100, 1e-10);
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
    
    private void worstStateAlphas(POMDP<Double> pomdp, MDPRewards<Double> mdpRewards) {
      
      for (int state = 0 ; state < pomdp.getNumStates() ; state++) {
        for (int action = 0 ; action < pomdp.getNumChoices(state); action++) {
          Object actionName = pomdp.getAction(state, action);
          if (!alphaVectors.containsKey(actionName)) {
            alphaVectors.put(actionName, new Double[pomdp.getNumStates()]);
            Arrays.fill(alphaVectors.get(actionName), 0.0);
          }
        }
      }
      
      
      
      for (Object actionName: alphaVectors.keySet()) {
        
        for (int state = 0 ; state < pomdp.getNumStates() ; state++) {
          int action = pomdp.getChoiceByAction(state, actionName);
          
          if (action == -1)
            continue; // action not possible from state
          
          alphaVectors.get(actionName)[state] = mdpRewards.getTransitionReward(state, action);
          
        }
      }
      
    }
    
    private void update(POMDP<Double> pomdp, MDPRewards<Double> mdpRewards) {
      Double[] alphaTemporary = new Double[pomdp.getNumStates()];
      Arrays.fill(alphaTemporary, 0.0);
      
      for (Object actionName: alphaVectors.keySet()) {
        
        for (int state = 0 ; state < pomdp.getNumStates() ; state++) {
          int action = pomdp.getChoiceByAction(state, actionName);
          
          if (action == -1) {
            alphaTemporary[state] = 0.0;
            continue; // action not possible from state
          }
          
          
          Double reward = mdpRewards.getTransitionReward(state, action);
          
          Double value = 0.0;
          
          double[] certainStateBelief = new double[pomdp.getNumStates()];
          certainStateBelief[state] = 1.0;
          double[] successor = pomdp.getBeliefInDistAfterChoice(certainStateBelief, action);
          

          for (int succState = 0 ; succState < pomdp.getNumStates() ; succState++) {
            if (successor[succState] != 0.0) {
              value += successor[succState] * alphaVectors.get(actionName)[succState];
            }  
          }
          
          alphaTemporary[state] = value + reward;
          
        }

        residuals.put(actionName, computeBeliefResiduals(alphaVectors.get(actionName), alphaTemporary));
        alphaVectors.put(actionName, alphaTemporary.clone());
        
      }
      
    }
    
    public HashMap<Object, Double[]> computePolicy(POMDP<Double> pomdp, MDPRewards<Double> mdpRewards){
      worstStateAlphas(pomdp, mdpRewards); // this initialises alphaVectors
      
      residuals = new HashMap<Object, Double>();
      
      for (int state = 0 ; state < pomdp.getNumStates() ; state++) {
        for (int action = 0 ; action < pomdp.getNumChoices(state); action++) {
          Object actionName = pomdp.getAction(state, action);
          residuals.put(actionName, 0.0);
        }
      }
      
      long t0 = System.currentTimeMillis();
      
      int iter = 0;
      while (iter < maxIterations && (System.currentTimeMillis() - t0) / 1000.0 < maxTime) {
        update(pomdp, mdpRewards);
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


