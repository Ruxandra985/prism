package hsvirp;

import java.util.AbstractMap;
import java.util.Arrays;
import java.util.BitSet;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import explicit.POMDP;
import explicit.rewards.MDPRewards;

public class SightedPolicyLowerBound {
    private int maxIterations;
    private double maxTime;
    private double belRes;
    private Set<Map.Entry<Object, Double>> residuals;
    private Set<Object> addedActions = new HashSet<>();

    Set<Map.Entry<Object, Double[]>> alphaVectors = new HashSet<>();

    Set<Map.Entry<Object, Double[]>> alphaVectorsNew = new HashSet<>();
    
    // Note: discount factor is taken as 1
    
    // Constructor
    public SightedPolicyLowerBound(int maxIter, double maxTime, double beliefResidual) {
        this.maxIterations = maxIter;
        this.maxTime = maxTime;
        this.belRes = beliefResidual;
        this.residuals = new HashSet<>();
        
    }

    // Default Constructor
    public SightedPolicyLowerBound() {
        this(10, 100, 1e-10);
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
    
    private boolean isDominated (POMDP<Double> pomdp, Set<Map.Entry<Object, Double[]>> dominator, Double[] alphaTemporary, boolean equalDominates) {
      int equalCounts = 0;
      
      for (Map.Entry<Object, Double[]> alphaVec : dominator) {
        Double[] vec = alphaVec.getValue();
        
        boolean dominated = true;
        
        boolean equal = true;
        
        for (int s = 0 ; s < pomdp.getNumStates() ; s++) {
          
          if (!alphaTemporary[s].equals(vec[s]))
            equal = false;
          
          if (alphaTemporary[s] > vec[s] + 1e-4) {
            dominated = false;
            break;
          }
        }
        
        if (equal)
          equalCounts++;
        
        //if (equalCounts > 1)
          //System.out.println("clones");
        
        
        if (equalDominates) {
          if (equal || dominated)
            return true;
        }
        else if (dominated && !equal)
          return true;
        
      }
      return false;
    }
    
    private void worstStateAlphas(POMDP<Double> pomdp, MDPRewards<Double> mdpRewards, BitSet remain) {
      
      for (int state = 0 ; state < pomdp.getNumStates() ; state++) {
        for (int action = 0 ; action < pomdp.getNumChoices(state); action++) {
          Object actionName = pomdp.getAction(state, action);
          if (!addedActions.contains(actionName)) {
            addedActions.add(actionName);
          }
        }
      }
      
      
      
      for (Object actionName : addedActions) {
        
        Double[] arrayInit = new Double[pomdp.getNumStates()];
        Arrays.fill(arrayInit, 0.0);
        
        for (int state = 0 ; state < pomdp.getNumStates() ; state++) {
          int action = pomdp.getChoiceByAction(state, actionName);
          
          if (action == -1 || (remain != null && !remain.get(state))) 
            continue; // action not possible from state
          
          arrayInit[state] = mdpRewards.getTransitionReward(state, action);
          
        }
        alphaVectors.add(new AbstractMap.SimpleEntry<>(actionName, arrayInit));
      }
      
      alphaVectorsNew = alphaVectors; // initially
      
    }
    
    private void update(POMDP<Double> pomdp, MDPRewards<Double> mdpRewards, BitSet remain) {
      Double[] alphaTemporary = new Double[pomdp.getNumStates()];
      Arrays.fill(alphaTemporary, 0.0);
      
      Set<Map.Entry<Object, Double[]>> alphaVectorsToAdd = new HashSet<>();
      
      for (Map.Entry<Object, Double[]> alphaVec : alphaVectors) {
        Object actionName = alphaVec.getKey();
        for (Map.Entry<Object, Double[]> alphaVecNext : alphaVectorsNew) {
        
          Double[] vec = alphaVecNext.getValue();
          
          Double sumNewArr = 0.0;
          
          for (int state = 0 ; state < pomdp.getNumStates() ; state++) {
            int action = pomdp.getChoiceByAction(state, actionName);
            
            if (action == -1 || (remain != null && !remain.get(state))) {
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
                value += successor[succState] * vec[succState];
              }  
            }
            
            alphaTemporary[state] = value + reward;
            
            sumNewArr += value + reward;
          }
          
          if (sumNewArr > 0.0 && !isDominated(pomdp, alphaVectors, alphaTemporary, true) && !isDominated(pomdp, alphaVectorsToAdd, alphaTemporary, true)) {
            // not arrays full of zeroes and check it is not dominated by any others
            residuals.add(new AbstractMap.SimpleEntry<>(actionName, computeBeliefResiduals(alphaVec.getValue(), alphaTemporary)));
            alphaVectorsToAdd.add(new AbstractMap.SimpleEntry<>(actionName, alphaTemporary.clone()));
          }
          
        }
      }
      
      // do some pruning for alphaVectors
      
      
      Set<Map.Entry<Object, Double[]>> alphaVectorsToRemove = new HashSet<>();
      //System.out.println();
      for (Map.Entry<Object, Double[]> alphaVec : alphaVectors) {
        if (isDominated(pomdp, alphaVectors, alphaVec.getValue(), false) || isDominated(pomdp, alphaVectorsToAdd, alphaVec.getValue(), false))
          alphaVectorsToRemove.add(alphaVec);
      }
      
      for (Map.Entry<Object, Double[]> alphaVec : alphaVectorsToAdd) {
        alphaVectors.add(alphaVec);
      }
      
      for (Map.Entry<Object, Double[]> alphaVec : alphaVectorsToRemove) {
        alphaVectors.remove(alphaVec);
      }
      
      alphaVectorsNew = alphaVectorsToAdd;
      
      
      
    }
    
    public Set<Map.Entry<Object, Double[]>> computePolicy(POMDP<Double> pomdp, MDPRewards<Double> mdpRewards, BitSet remain){
      worstStateAlphas(pomdp, mdpRewards, remain); // this initialises alphaVectors
      
      residuals = new HashSet<>();
      
      for (int state = 0 ; state < pomdp.getNumStates() ; state++) {
        for (int action = 0 ; action < pomdp.getNumChoices(state); action++) {
          Object actionName = pomdp.getAction(state, action);
          residuals.add(new AbstractMap.SimpleEntry<>(actionName, 0.0));
        }
      }
      
      long t0 = System.currentTimeMillis();
      
      int iter = 0;
      while (iter < maxIterations && (System.currentTimeMillis() - t0) / 1000.0 < maxTime) {
        iter++;
        //System.out.println(iter);
        update(pomdp, mdpRewards, remain);
        
        boolean smallerThanBelRes = true;
        
        for (Map.Entry<Object, Double> residual: residuals) {
          smallerThanBelRes = smallerThanBelRes && (residual.getValue() < belRes);
        }
        
        if (smallerThanBelRes || alphaVectorsNew.isEmpty()) 
            break;
      }
      
      
      return alphaVectors;
    }
    
}


