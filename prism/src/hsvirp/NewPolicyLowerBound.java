package hsvirp;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.BitSet;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.SortedSet;

import explicit.POMDP;
import explicit.rewards.MDPRewards;

public class NewPolicyLowerBound {
  private int maxIterations;
  private double maxTime;
  private double belRes;
  private HashMap<Object, Double> residuals;
  private List<Integer> immRwdMaximiser;
  private int maxNrActionsFromState = 0;

  HashMap<Integer, Double[]> alphaVectors = new HashMap<>();
  
  // Note: discount factor is taken as 1
  
  // Constructor
  public NewPolicyLowerBound(int maxIter, double maxTime, double beliefResidual) {
      this.maxIterations = maxIter;
      this.maxTime = maxTime;
      this.belRes = beliefResidual;
      this.residuals = new HashMap<>();
      this.immRwdMaximiser = new ArrayList<>();
  }

  // Default Constructor
  public NewPolicyLowerBound() {
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
  
  private void worstStateAlphas(POMDP<Double> pomdp, MDPRewards<Double> mdpRewards, BitSet remain) {
    
    // what if you got the total nr of actions
    for (int state = 0 ; state < pomdp.getNumStates() ; state++) {
      maxNrActionsFromState = Math.max(maxNrActionsFromState, pomdp.getNumChoices(state));
      Double maxRew = 0.0;
      int maxRewAction = 0;
      for (int action = 0 ; action < pomdp.getNumChoices(state); action++) {
        if (!alphaVectors.containsKey(action)) {
          alphaVectors.put(action, new Double[pomdp.getNumStates()]);
          Arrays.fill(alphaVectors.get(action), 0.0);
        }
        if (maxRew <= mdpRewards.getTransitionReward(state, action)) {
          maxRew = mdpRewards.getTransitionReward(state, action);
          maxRewAction = action;
        }
      }
      immRwdMaximiser.add(maxRewAction); // action that maximises imm reward for the state
    }
    
    
    
    for (Integer actionPosition: alphaVectors.keySet()) {
      
      // take the actionPosition th action from each state
      
      for (int state = 0 ; state < pomdp.getNumStates() ; state++) {
        if (remain != null && !remain.get(state))
            continue;
        
        if (actionPosition < pomdp.getNumChoices(state))
          alphaVectors.get(actionPosition)[state] = mdpRewards.getTransitionReward(state, actionPosition);
        else {
          // just take the action here that maximises the immediate reward - a greedy approach
          alphaVectors.get(actionPosition)[state] = mdpRewards.getTransitionReward(state, immRwdMaximiser.get(state));
        }
        
      }
    }
    
  }
  
  private void update(POMDP<Double> pomdp, MDPRewards<Double> mdpRewards, BitSet remain) {
    Double[] alphaTemporary = new Double[pomdp.getNumStates()];
    Arrays.fill(alphaTemporary, 0.0);
    
    for (Integer actionPosition: alphaVectors.keySet()) {
      
      for (int state = 0 ; state < pomdp.getNumStates() ; state++) {
        
        if ((remain != null && !remain.get(state))) {
          alphaTemporary[state] = 0.0;
          continue; // not a good state
        }
        
        int action = actionPosition < pomdp.getNumChoices(state) ? actionPosition : immRwdMaximiser.get(state);
        
        Double reward = mdpRewards.getTransitionReward(state, action);
        
        Double value = 0.0;
        
        double[] certainStateBelief = new double[pomdp.getNumStates()];
        certainStateBelief[state] = 1.0;
        double[] successor = pomdp.getBeliefInDistAfterChoice(certainStateBelief, action);
        

        for (int succState = 0 ; succState < pomdp.getNumStates() ; succState++) {
          if (successor[succState] != 0.0) {
            value += successor[succState] * alphaVectors.get(actionPosition)[succState];
          }  
        }
        
        alphaTemporary[state] = value + reward;
        
      }

      residuals.put(actionPosition, computeBeliefResiduals(alphaVectors.get(actionPosition), alphaTemporary));
      alphaVectors.put(actionPosition, alphaTemporary.clone());
      
    }
    
  }
  
  public HashMap<Object, Double[]> computePolicy(POMDP<Double> pomdp, MDPRewards<Double> mdpRewards, BitSet remain){
    worstStateAlphas(pomdp, mdpRewards, remain); // this initialises alphaVectors
    
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
      update(pomdp, mdpRewards, remain);
      iter++;
      
      boolean smallerThanBelRes = true;
      
      for (Double residual: residuals.values()) {
        smallerThanBelRes = smallerThanBelRes && (residual < belRes);
      }
      
      if (smallerThanBelRes) 
          break;
    }
    
    HashMap<Object, Double[]> resultAlphaVec = new HashMap<>();
    for (Map.Entry<Integer, Double[]> entry: alphaVectors.entrySet()) {
      resultAlphaVec.put(entry.getKey(), entry.getValue());
    }
    return resultAlphaVec;
  }
}
