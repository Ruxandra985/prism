package hsvirp;

import java.util.AbstractMap;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.BitSet;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import explicit.POMDP;
import explicit.rewards.MDPRewards;

public class SightedPolicyLowerBound {
    private int maxIterations;
    private double maxTime;
    private double belRes;
    private HashMap<List<Object>, Double> residuals;
    private Set<Object> addedActions = new HashSet<>();

    HashMap<List<Object>, Double[]> alphaVectors = new HashMap<>();

    HashMap<List<Object>, Double[]> alphaVectorsNew = new HashMap<>();
    
    // Note: discount factor is taken as 1
    
    // Constructor
    public SightedPolicyLowerBound(int maxIter, double maxTime, double beliefResidual) {
        this.maxIterations = maxIter;
        this.maxTime = maxTime;
        this.belRes = beliefResidual;
        this.residuals = new HashMap<>();
        
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
    
    private boolean isDominated (POMDP<Double> pomdp, HashMap<List<Object>, Double[]> dominator, Double[] alphaTemporary, boolean equalDominates) {
      int equalCounts = 0;
      
      for (Map.Entry<List<Object>, Double[]> alphaVec : dominator.entrySet()) {
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
        
        double arraySum = 0.0;
        
        for (int state = 0 ; state < pomdp.getNumStates() ; state++) {
          int action = pomdp.getChoiceByAction(state, actionName);
          
          if (action == -1 || (remain != null && !remain.get(state))) 
            continue; // action not possible from state
          
          arrayInit[state] = mdpRewards.getTransitionReward(state, action);
          
          arraySum += arrayInit[state];
          
        }
        
        if (arraySum > 0.0) {
          // do not consider vectors that are just full of zeroes
          List<Object> actionsIdentified = new ArrayList<>();
          actionsIdentified.add(actionName);
          
          alphaVectorsNew.put(actionsIdentified, arrayInit);
        }
      }
      
      
    }
    
    private void update(POMDP<Double> pomdp, MDPRewards<Double> mdpRewards, BitSet remain) {
      Double[] alphaTemporary = new Double[pomdp.getNumStates()];
      Arrays.fill(alphaTemporary, 0.0);
      
      HashMap<List<Object>, Double[]> alphaVectorsToAdd = new HashMap<>();
      HashMap<List<Object>, Double[]> alphaVectorsToExtend = new HashMap<>();
      
      // first we do lookahead step on all of the old vectors
      for (Map.Entry<List<Object>, Double[]> alphaVec : alphaVectors.entrySet()) {
        
        List<Object> actionsIdentified = alphaVec.getKey();
        // we take actions based on order in actionsIdentified list.
        // If some states have none of the actionsIdentified, we add one alpha vec
        // for each action we can augment our list with
        
        Double[] vec = alphaVec.getValue();
          
        Double sumNewArr = 0.0;
          
        for (int state = 0 ; state < pomdp.getNumStates() ; state++) {
          
          if (remain != null && !remain.get(state)){
            alphaTemporary[state] = 0.0;
            continue; // need to ignore state
          }
          
          boolean foundAvailable = false;
          Object actionFound = null;
          for (Object knownAction : actionsIdentified) {
            int action = pomdp.getChoiceByAction(state, knownAction);
            
            if (action != -1){
              foundAvailable = true;
              actionFound = knownAction;
              break;
            }
          }
          
          if (foundAvailable) { // we take that action, as per our policy
            int action = pomdp.getChoiceByAction(state, actionFound);
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
          else {
            // this set belongs in S
            alphaTemporary[state] = 0.0;
          }
            
          
        }
        
        if (sumNewArr == 0.0)
          continue; // irrelevant array
        
        // ok, did one step lookahead
        alphaVectorsToAdd.put(actionsIdentified, alphaTemporary.clone());
        residuals.put(actionsIdentified, computeBeliefResiduals(vec, alphaTemporary));
        // we claim these arrays have been extended before at some point if
        // extension was possible, so we don't extend them anymore or in the future
      }
      
      for (Map.Entry<List<Object>, Double[]> alphaVec : alphaVectorsNew.entrySet()) {
        
        List<Object> actionsIdentified = alphaVec.getKey();
        // we take actions based on order in actionsIdentified list.
        // If some states have none of the actionsIdentified, we add one alpha vec
        // for each action we can augment our list with
        
        Double[] vec = alphaVec.getValue();
          
        Double sumNewArr = 0.0;
        
        ArrayList<Integer> uncertainStates = new ArrayList<>();
          
        for (int state = 0 ; state < pomdp.getNumStates() ; state++) {
          
          if (remain != null && !remain.get(state)){
            alphaTemporary[state] = 0.0;
            continue; // need to ignore state
          }
          
          boolean foundAvailable = false;
          Object actionFound = null;
          for (Object knownAction : actionsIdentified) {
            int action = pomdp.getChoiceByAction(state, knownAction);
            
            if (action != -1){
              foundAvailable = true;
              actionFound = knownAction;
              break;
            }
          }
          
          if (foundAvailable) { // we take that action, as per our policy
            int action = pomdp.getChoiceByAction(state, actionFound);
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
          else {
            // this set belongs in S
            alphaTemporary[state] = 0.0;
            uncertainStates.add(state);
          }
            
          
        }
        
        if (sumNewArr == 0.0)
          continue; // irrelevant array
        
        // ok. You still need to lookahead for new vectors too 
        alphaVectorsToAdd.put(actionsIdentified, alphaTemporary.clone());
        residuals.put(actionsIdentified, computeBeliefResiduals(vec, alphaTemporary));
        // but in this case you also try to extend, but not in the future, just now
        if (!uncertainStates.isEmpty()) {
          for (Object actionName: addedActions) {
            if (!actionsIdentified.contains(actionName)) {
              // enhance our alpha vector by allowing the uncertain states to take actionName if they can
              
              List<Object> newActId = (List<Object>)((ArrayList<Object>) actionsIdentified).clone();
              newActId.add(actionName);
              
              if (alphaVectorsToAdd.containsKey(newActId))
                continue; // already contains 
              
              boolean enhanced = false;
              
              for (Integer state : uncertainStates) {
                int action = pomdp.getChoiceByAction(state, actionName);
                if (action == -1) { // action not available
                  alphaTemporary[state] = 0.0;
                  continue;
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
                if (alphaTemporary[state] > 0.0)
                  enhanced = true; // it enhanced at least one previously uncertain State
              }
              
              
              if (enhanced) {
                // we have obtained one new alpha vector!
                
                alphaVectorsToExtend.put(newActId, alphaTemporary.clone());
                residuals.put(newActId, computeBeliefResiduals(vec, alphaTemporary));
              }
              
            }
            
          }
        }
          
      }
      
      // do some pruning for alphaVectors maybe
      
      // get rid of the previous vectors.
      
      Set<List<Object>> alphaVectorsToRemove = new HashSet<>();
      
      for (Map.Entry<List<Object>, Double[]> alphaVec : alphaVectorsToAdd.entrySet()) {
        if (isDominated(pomdp, alphaVectorsToAdd, alphaVec.getValue(), false))
          alphaVectorsToRemove.add(alphaVec.getKey());
        // see if this is dominated  
      }
      
      // the vectors that have ever been extended we will not try to extend later
      // we only look at extending the newly extended ones
      alphaVectors = (HashMap<List<Object>, Double[]>)alphaVectorsToAdd.clone();
      
      alphaVectorsNew = (HashMap<List<Object>, Double[]>)alphaVectorsToExtend.clone();
      
      
    }
    
    public HashMap<Object, Double[]> computePolicy(POMDP<Double> pomdp, MDPRewards<Double> mdpRewards, BitSet remain){
      worstStateAlphas(pomdp, mdpRewards, remain); // this initialises alphaVectors
      
      for (int state = 0 ; state < pomdp.getNumStates() ; state++) {
        for (int action = 0 ; action < pomdp.getNumChoices(state); action++) {
          Object actionName = pomdp.getAction(state, action);
          List<Object> actionsIdentified = new ArrayList<>();
          actionsIdentified.add(actionName);
          residuals.put(actionsIdentified, 0.0);
        }
      }
      
      long t0 = System.currentTimeMillis();
      
      int iter = 0;
      while (iter < maxIterations && (System.currentTimeMillis() - t0) / 1000.0 < maxTime) {
        iter++;
        update(pomdp, mdpRewards, remain);

        boolean smallerThanBelRes = true;
        
        for (Map.Entry<List<Object>, Double> residual: residuals.entrySet()) {
          smallerThanBelRes = smallerThanBelRes && (residual.getValue() < belRes);
        }
        
        if (smallerThanBelRes) 
            break;
      }
      
      Set<List<Object>> alphaVectorsToRemove = new HashSet<>();
      
      for (Map.Entry<List<Object>, Double[]> alphaVec : alphaVectors.entrySet()) {
        if (isDominated(pomdp, alphaVectors, alphaVec.getValue(), false))
          alphaVectorsToRemove.add(alphaVec.getKey());
        // see if this is dominated  
      }
      
      
      // we will aim to remove duplicates from alphaVectors
      
      Map<List<Object>, Double[]> noDuplAlphaVec = new HashMap<>();
      
      
      for (Map.Entry<List<Object>, Double[]> entry1: alphaVectors.entrySet()) {
        // add entry1 to the no duplicates map if you can
        
        boolean canAdd = true;
        
        for (Map.Entry<List<Object>, Double[]> entry2: noDuplAlphaVec.entrySet()) {
          if (Arrays.equals(entry2.getValue(), entry1.getValue()) 
              && !entry1.getKey().equals(entry2.getKey())) {
            canAdd = false;
            break;
          }
        }
        
        if (canAdd) {
          noDuplAlphaVec.put(entry1.getKey(), entry1.getValue());
        }
      }
      
      HashMap<Object, Double[]> resultAlphaVec = new HashMap<>();
      for (Map.Entry<List<Object>, Double[]> entry: noDuplAlphaVec.entrySet()) {
        resultAlphaVec.put(entry.getKey(), entry.getValue());
      }
      return resultAlphaVec;
    }
    
}


