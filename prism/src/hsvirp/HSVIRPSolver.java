package hsvirp;

import java.util.AbstractMap;
import java.util.ArrayList;

import prism.Accuracy;
import prism.Accuracy.AccuracyLevel;
import java.util.Arrays;
import java.util.BitSet;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.Map.Entry;
import java.util.Random;
import java.util.stream.IntStream;
import explicit.Belief;
import explicit.ModelCheckerResult;
import explicit.POMDP;
import explicit.rewards.MDPRewards;
import prism.PrismException;

public class HSVIRPSolver {
    public BlindPolicyLowerBound lowerBound;
    public FastInformedUpperBound upperBound;
    public SightedPolicyLowerBound spol;
    private HSVITree tree;
    private POMDP<Double> pomdp;
    private MDPRewards<Double> mdpRewards;
    
    private double epsilon = 0.01;
    private double precision = 1e-3;
    private double kappa = 0.5;
    private double delta = 0.01;
    private long maxTime = 30000000; // only max 10 seconds
    private int maxSteps = Integer.MAX_VALUE;
    private int initSteps = 200;
    private int stepIncrement = 10;
    private double pruneThreshold = 0.10;
    // Note: discount factor is taken as 1
    private int nrIterations = 10;
    private HashMap<Integer, Integer> cycleDetector = new HashMap<>();
    private Set<Integer> actionsFromCurr = new HashSet<>(); // this is here just to be shared between methods
    private HashMap<Integer, HashMap<Integer, HashMap<Object, Double>>> alphaObs = new HashMap<>();
    
    // Constructor
    public HSVIRPSolver() {
      this.lowerBound = new BlindPolicyLowerBound();
      this.upperBound = new FastInformedUpperBound();
      this.spol = new SightedPolicyLowerBound();
      
    }
    
    public Set<Map.Entry<Object, Double[]>> computeLowerBoundPolicy(POMDP<Double> pomdp, MDPRewards<Double> mdpRewards, BitSet remain){
      //return spol.computePolicy(pomdp, mdpRewards, remain);
      return lowerBound.computePolicy(pomdp, mdpRewards).entrySet();
    }
    
    public Set<Map.Entry<Object, Double[]>> computeUpperBoundPolicy(POMDP<Double> pomdp, MDPRewards<Double> mdpRewards, BitSet target, BitSet remain){
      return upperBound.computePolicy(pomdp, mdpRewards, target, remain).entrySet();
    }

    public ModelCheckerResult solve (POMDP<Double> pomdp, BitSet target, BitSet remain, boolean min, int sInit, MDPRewards<Double> mdpRewards) throws PrismException{
      
      
      //SightedPolicyLowerBound spol = new SightedPolicyLowerBound();
      
      //spol.computePolicy(pomdp, mdpRewards);
      
      
      int valueIterMax = 10000;
      
      this.pomdp = pomdp;
      
      this.mdpRewards = mdpRewards;
      
      // computes upper and lower bounds.
      
      this.tree = HSVITree.initializeTree(this, pomdp, target, remain, mdpRewards);
      // initialising tree also introduces initial belief state as the root and in the frontier
      
      int indexOfRoot = 0;
      
      int depthTrial = initSteps + stepIncrement;
      
      int numAlphas = tree.VsLowerBound.size();
      
      long t0 = System.currentTimeMillis();
      
      int nrOverallIterations = 0;
      
      double effectiveDiscount = 0.999999;
      
      while (depthTrial < maxSteps &&
          System.currentTimeMillis() - t0 < maxTime &&
          tree.VUpper.get(indexOfRoot) - tree.VLower.get(indexOfRoot) > precision) {
        
        nrOverallIterations++;
        
        int nrSubIter = 0;
        double prevRootDiff = tree.VUpper.get(indexOfRoot) - tree.VLower.get(indexOfRoot);
        

        long currTime = (System.currentTimeMillis() - t0);
        double rootDiff = tree.VUpper.get(indexOfRoot) - tree.VLower.get(indexOfRoot); 
        
        System.out.println("time: " + currTime +
            " diff: " + rootDiff +
            " lower: " + tree.VLower.get(indexOfRoot) +
            " upper: " + tree.VUpper.get(indexOfRoot) +
            " beliefs: " + tree.beliefNodes.size() + 
            " alphaVec " + tree.VsLowerBound.size() +
            " depth trial: " + depthTrial);
        
        while (tree.VUpper.get(indexOfRoot) - tree.VLower.get(indexOfRoot) > precision &&
            System.currentTimeMillis() - t0 < maxTime &&
            nrSubIter < nrIterations) {
          // sample
          sample(depthTrial, effectiveDiscount);
          // backup
          backupFrontier(); // should ensure the new bounds are indeed computed
          nrSubIter++;
          // here check for pruning if needed
          
          if (tree.VsLowerBound.size() > 1.1 * numAlphas) {
            prune();
            numAlphas = tree.VsLowerBound.size();
          }
          currTime = (System.currentTimeMillis() - t0);
          rootDiff = tree.VUpper.get(indexOfRoot) - tree.VLower.get(indexOfRoot); 
          System.out.println("time: " + currTime +
              " diff: " + rootDiff +
              " lower: " + tree.VLower.get(indexOfRoot) +
              " upper: " + tree.VUpper.get(indexOfRoot) +
              " beliefs: " + tree.beliefNodes.size() + 
              " alphaVec " + tree.VsLowerBound.size() +
              " depth trial: " + depthTrial);
          
          
        }
        
        currTime = (System.currentTimeMillis() - t0);
        rootDiff = tree.VUpper.get(indexOfRoot) - tree.VLower.get(indexOfRoot); 
        System.out.println("time: " + currTime +
            " diff: " + rootDiff +
            " lower: " + tree.VLower.get(indexOfRoot) +
            " upper: " + tree.VUpper.get(indexOfRoot) +
            " beliefs: " + tree.beliefNodes.size() + 
            " alphaVec " + tree.VsLowerBound.size() );
        
        if (Math.abs(tree.VUpper.get(indexOfRoot) - tree.VLower.get(indexOfRoot) - prevRootDiff) < 1e-2) {
          depthTrial += stepIncrement;
          effectiveDiscount += 9 / Math.pow(10, Double.toString(effectiveDiscount).length() - 1);
        }
        
        // Update V_upper for beliefs with no children
        for (int beliefIndex = 0 ; beliefIndex < tree.beliefNodes.size() ; beliefIndex++) {
          List<Double> vUpperCopy = List.copyOf(tree.VUpper);
          if (!tree.beliefNodeHasChildren.get(beliefIndex)) {
            tree.VUpper.set(beliefIndex, resetUpper(vUpperCopy, tree.beliefNodes.get(beliefIndex)));
          }
        }
        
        // Update V_upper for beliefs with children
        for (int beliefIndex = 0 ; beliefIndex < tree.beliefNodes.size() ; beliefIndex++) {
          if (tree.beliefNodeHasChildren.get(beliefIndex)) {
            tree.VUpper.set(beliefIndex, tree.VLower.get(beliefIndex));
          }
        }
        
        // Iterate to adjust V_upper until no changes occur or maximum iterations reached
        boolean changed = true;
        for (int i = 0; i < valueIterMax && changed; i++) {
          changed = false;
            // Iterate in reverse over beliefs
            for (int beliefIndex = tree.beliefNodes.size() - 1 ; beliefIndex >= 0 ; beliefIndex--) {
                if (tree.beliefNodeHasChildren.get(beliefIndex)) {
                    double oldValue = tree.VUpper.get(beliefIndex);
                    backupUpper(beliefIndex); // Update V_upper for the belief
                    double newValue = tree.VUpper.get(beliefIndex);

                    // Check if there is a significant change in the value
                    if (Math.abs(oldValue - newValue) > 1e-10) {
                        changed = true;
                    }
                }
            }

            // if no changes, for loop finishes
        }
      }
      
      Double hi = tree.VUpper.get(indexOfRoot);
      Double lo = tree.VLower.get(indexOfRoot);
      System.out.println("Result bounds: [" + lo + "," + hi + "]");
      double soln[] = new double[pomdp.getNumStates()];
      
      soln[sInit] = tree.VLower.get(indexOfRoot);
      
      
      Double err = (hi - lo) / 2.0;
      
      
      ModelCheckerResult res = new ModelCheckerResult();
      
      res.numIters = nrOverallIterations;
      res.soln = soln;
      res.accuracy = new Accuracy(err == 0.0 ? AccuracyLevel.EXACT_FLOATING_POINT : AccuracyLevel.BOUNDED, err, true); 
      
      return res; 
    }
    
    private Double resetUpper(List<Double> vUpperCopy, Belief belief) {
      // Retrieve corner alpha vectors and compute VCorner
      Double[] cornerAlphas = new Double[tree.VsUpperBound.size()];
      tree.VsUpperBound.toArray(cornerAlphas);
      double[] beliefDist = belief.toDistributionOverStates(tree.pomdp);
      double vCorner = HSVITree.dot(cornerAlphas, beliefDist);

      double vMin = Double.POSITIVE_INFINITY;
      
      for (int beliefIndex : tree.real) {
        Belief beliefNodeIndex = tree.beliefNodes.get(beliefIndex);
        
        
        if (HSVITree.isTerminal(tree, beliefNodeIndex)) 
          continue;
        
        double vUpperNow = vUpperCopy.get(beliefIndex);
        
        double[] beliefNodeIndexDist = beliefNodeIndex.toDistributionOverStates(pomdp);
        double phi = Double.POSITIVE_INFINITY;
        
        
        
        for (int i = 0 ; i < beliefDist.length ; i++) {
          if (beliefDist[i] != 0.0 && beliefNodeIndexDist[i] != 0.0) {
            if (phi > beliefDist[i] / beliefNodeIndexDist[i]) 
              phi = beliefDist[i] / beliefNodeIndexDist[i];
          }
          else if (beliefDist[i] == 0.0 && beliefNodeIndexDist[i] != 0.0) {
            phi = 0.0;
            break;
          }
        }
        
        double vHat = vCorner + phi * (vUpperNow - HSVITree.dot(cornerAlphas, beliefNodeIndexDist));
        if (vHat < vMin) {
          vMin = vHat;
        }
      }
      
      return vMin;
    }

    public void sample(int depthTrial, double effectiveDiscount) {
      tree.frontier.clear();
      cycleDetector.clear(); // reset the cycleDetector?
      
      int indexOfRoot = 0;
      double rootDiff = tree.VUpper.get(indexOfRoot) - tree.VLower.get(indexOfRoot);
      samplePoints(indexOfRoot, 0, epsilon * rootDiff , depthTrial, effectiveDiscount);
    }

    private void samplePoints(int currNode, int currDepth, double difference, int depthTrial, double effectiveDiscount) {
      
      tree.bPruned.set(currNode, false);
      
      Belief beliefState = tree.beliefNodes.get(currNode);
      
      if (tree.real.indexOf(currNode) == -1) { // currNode is not in the real nodes list
        tree.real.add(currNode); // add it to real nodes
      }
      
      if (HSVITree.isTerminal(tree, beliefState)) {
        return;
      }
      
      // we are not in a target belief state
      HSVITree.fillBelief(tree, currNode);
      
      double VLower = tree.VLower.get(currNode);
      double VUpper = tree.VUpper.get(currNode);
      
      if (VUpper <= VLower + difference * kappa * Math.pow(effectiveDiscount, -currDepth) || currDepth > depthTrial ) {
        tree.frontier.add(currNode);
        return;
      }
      
      // now we must use our chosen heuristics to pick an action and observation
      
      int[] actionObservation = chooseActionAndObservation(currNode, currDepth + 1);
      int action = actionObservation[0];
      int observationIdx = actionObservation[1];
      
      if (action < 0 || observationIdx < 0) { // something failed in choosing action and observation
        tree.frontier.add(currNode);
        cycleDetector.put(currNode, -1);
        if (cycleDetector.values().stream().allMatch(v -> v == -1)) { // all values are marked with -1
          return;
        }

        for (int frontierNode : tree.frontier) {
          if (cycleDetector.containsKey(frontierNode) && cycleDetector.get(frontierNode) != -1) { // not marked as invalid
              samplePoints(frontierNode, currDepth + 1, difference, depthTrial, effectiveDiscount); 
              // you get a different expandable node and expand it instead
              return;
          }
        }
      }
      
      // otherwise you continue
      tree.beliefSampleFrequency.set(currNode, tree.beliefSampleFrequency.get(currNode) + 1); // increments frequency
      
      tree.beliefActionSampleFrequency.get(currNode).set(action, 
          tree.beliefActionSampleFrequency.get(currNode).get(action) + 1); // increments frequency
      
      tree.beliefActObsSampleFrequency.get(currNode).get(action).set(observationIdx, 
          tree.beliefActObsSampleFrequency.get(currNode).get(action).get(observationIdx) + 1); // increments frequency
           
      tree.frontier.add(currNode);
      
      Integer childNode = tree.beliefAndActionChildren.get(currNode).get(action).get(observationIdx);
      
      samplePoints(childNode, currDepth + 1, difference, depthTrial, effectiveDiscount);
      
    }
    
    private int[] chooseActionAndObservation(int currNode,  int currDepth) {
      Belief beliefState = tree.beliefNodes.get(currNode);
      int nrActions = tree.pomdp.getNumChoicesForObservation(beliefState.so);
      Integer[] possibleActions = IntStream.range(0, nrActions).boxed().toArray(Integer[]::new);
      this.actionsFromCurr = new HashSet<>(Arrays.asList(possibleActions));
      // set with possible action choices you can take from currNode
      
      int action = -1;
      int observationIdx = -1;
      
      while (action < 0 || observationIdx < 0) {
        action = maxRandQ(currNode);
        
        if (action == -1) {
          break;
        }
        
        this.actionsFromCurr.remove(action);
        // pruning??
        tree.baPruned.get(currNode).set(action, false);
        observationIdx = bestObs(currNode, action, currDepth);
      }
      
      return new int[]{action, observationIdx};
    }

    private int bestObs(int currNode, int action, int currDepth) {
      double bestGap = -Double.MAX_VALUE;
      int bestObs = -1;
      
      // what observations can you have given current belief state and the taken action?
      Belief beliefState = tree.beliefNodes.get(currNode);
      double[] beliefDist = beliefState.toDistributionOverStates(pomdp);
      HashMap<Integer, Double> possibleObservations = tree.pomdp.computeObservationProbsAfterAction(beliefDist, action);
      
      int obsNr = -1;
      for (Map.Entry<Integer, Double> entry : possibleObservations.entrySet()) {
        obsNr++;
        Double obsProbability = entry.getValue();
        
        int childIndex = tree.beliefAndActionChildren.get(currNode).get(action).get(obsNr);
        
        if (tree.frontier.contains(childIndex)) {
          if (!this.cycleDetector.containsKey(childIndex))
            this.cycleDetector.put(childIndex, 1);
          else  
            continue;
        }
        
        if (tree.VUpper.get(childIndex) == 0.0) {
          continue; // cannot reach target states from child, continue
        }
        // is the way you get observations okay even??? - I think so
        double gap = tree.VUpper.get(childIndex) - tree.VLower.get(childIndex) + obsProbability * 
            (0.01 * Math.sqrt(tree.beliefSampleFrequency.get(currNode)) 
             / (1 + tree.beliefActObsSampleFrequency.get(currNode).get(action).get(obsNr)));
        if (gap > bestGap) {
          bestGap = gap;
          bestObs = obsNr;
        }
      }
      return bestObs;
    }
    
    private int maxRandQ(int currNode) {
      double QLower = Double.NEGATIVE_INFINITY;
      double QUpper = Double.NEGATIVE_INFINITY;
      int action = -1;
      
      double maxQaUpper = Collections.max(tree.QaUpper.get(currNode));
      List<Integer> toRemove = new ArrayList<>();

      for (int actionIter : actionsFromCurr) {
          double QUpperCurrent = tree.QaUpper.get(currNode).get(actionIter);
          double QLowerCurrent = tree.QaLower.get(currNode).get(actionIter);

          if (QLowerCurrent > QLower) {
              QLower = QLowerCurrent;
          }
          if (QUpperCurrent > QUpper) {
              QUpper = QUpperCurrent;
              action = actionIter;
          }
          if (Math.abs(QUpperCurrent - maxQaUpper) <= 0.1) {
            double maxVal = QUpperCurrent + 0.01 * 
                Math.sqrt(tree.beliefSampleFrequency.get(currNode)) / ( 1.0 + tree.beliefActionSampleFrequency.get(currNode).get(actionIter));
          
            if (maxVal > QUpper) {
              action = actionIter;
              QUpper = maxVal;
            }
          
          
          }
          
          if (QUpperCurrent < QLower)
            toRemove.add(action);
          
          
      }
      
      for (int i = 0 ; i < toRemove.size() ; i++) {
        this.actionsFromCurr.remove(toRemove.get(i));
      }
      
      return action;
    }


    private void backupFrontier() {
      for (int i = tree.frontier.size() - 1 ; i >= 0 ; i--) {
          backup(tree.frontier.get(i));
      }
    }
    
    private void backup (int beliefIndex) {
      Belief beliefState = tree.beliefNodes.get(beliefIndex);
      double[] beliefDist = beliefState.toDistributionOverStates(pomdp);
      
      int nrActions = tree.pomdp.getNumChoicesForObservation(beliefState.so);
      

      alphaObs.clear();
      
      for (int action = 0 ; action < nrActions ; action++) {
        HashMap<Integer, Double> possibleObservations = tree.pomdp.computeObservationProbsAfterAction(beliefDist, action);

        Object actionLabel = pomdp.getActionForObservation(beliefState.so, action);
        
        int obsNr = 0;
        for (Map.Entry<Integer, Double> entry : possibleObservations.entrySet()) {
          Integer observation = entry.getKey();
          int childIndex = tree.beliefAndActionChildren.get(beliefIndex).get(action).get(obsNr);
          Belief childBelief = tree.beliefNodes.get(childIndex);
          Double[] maxAlphaVal = getMaxAlphaVal(childBelief);
          
          for (int sIt = 0 ; sIt < maxAlphaVal.length; sIt++) {
            // before assigning to alphaObs, you must make sure it is holding the necessary info
            if (!alphaObs.containsKey(sIt))
              alphaObs.put(sIt, new HashMap<>());
            
            if (!alphaObs.get(sIt).containsKey(observation))
              alphaObs.get(sIt).put(observation, new HashMap<>());
            
            if (maxAlphaVal[sIt] == null)
              System.out.println("what???");
            
            alphaObs.get(sIt).get(observation).put(actionLabel, maxAlphaVal[sIt]);
          }
          
          
          
          obsNr++;
        }
      }
      
      Double[] bestAlphaVector = new Double[pomdp.getNumStates()];
      Object bestAction = null;
      double bestValue = Double.NEGATIVE_INFINITY;
      
      
      for (int action = 0 ; action < nrActions ; action++) {
        Object actionLabel = pomdp.getActionForObservation(beliefState.so, action);
        Double[] alphaAction = backupAlpha(actionLabel);
        double qValue = HSVITree.dot(alphaAction, beliefDist);
        tree.QaLower.get(beliefIndex).set(action, qValue);

        if (qValue > bestValue) {
            bestValue = qValue;
            bestAlphaVector = alphaAction.clone();
            bestAction = actionLabel;
        }
      }
      
      tree.VsLowerBound.add(new AbstractMap.SimpleEntry<>(bestAction, bestAlphaVector));
      tree.VLower.set(beliefIndex, bestValue); // updates the lower bound

      for (int action = 0 ; action < nrActions; action++) {
          double rewardNow = pomdp.getRewardAfterChoice(beliefState, action, tree.mdpRewards);
          tree.QaUpper.get(beliefIndex).set(action, rewardNow);

          HashMap<Integer, Double> possibleObservations = tree.pomdp.computeObservationProbsAfterAction(beliefDist, action);
          int obsNr = 0;
          for (Map.Entry<Integer, Double> entry : possibleObservations.entrySet()) {
              Double obsProbability = entry.getValue();
              int childIndex = tree.beliefAndActionChildren.get(beliefIndex).get(action).get(obsNr);
              double childUpperValue = tree.VUpper.get(childIndex);
              
              tree.QaUpper.get(beliefIndex).set(action, tree.QaUpper.get(beliefIndex).get(action) + obsProbability * childUpperValue);
              obsNr++;
          }
      }
      
      tree.VUpper.set(beliefIndex, Collections.max(tree.QaUpper.get(beliefIndex)));
      
    }
    
    private Double[] backupAlpha(Object actionLabel) {
      Double[] alpha = new Double[pomdp.getNumStates()];
      Arrays.fill(alpha, 0.0);
      for (int s = 0; s < pomdp.getNumStates(); s++) {
        double value = 0.0;
        int choiceNrFromS = pomdp.getChoiceByAction(s, actionLabel);
        
        if (tree.remain != null && !tree.remain.get(s))
          continue;
        
        if (choiceNrFromS != -1) {
          // you can take the given action from state s
          Iterator<Entry<Integer, Double>> transitionsIter = pomdp.getTransitionsIterator(s, choiceNrFromS);
            
          while(transitionsIter.hasNext()) {
            Entry<Integer, Double> e = transitionsIter.next();
            int stateInner = e.getKey();
            Double probabilityTransition = e.getValue();
            int obs = pomdp.getObservation(stateInner);
             
            if (alphaObs.containsKey(stateInner) &&
                alphaObs.get(stateInner).containsKey(obs) &&
                alphaObs.get(stateInner).get(obs).containsKey(actionLabel))
              value += probabilityTransition * alphaObs.get(stateInner).get(obs).get(actionLabel);
            
          }
          alpha[s] = pomdp.getRewardAfterChoice(stateToBeliefState(s), choiceNrFromS, mdpRewards) + value;
        }
      
      }
      return alpha;
    }
    
    private void backupUpper(int beliefIndex) {
      Belief beliefState = tree.beliefNodes.get(beliefIndex);
      double[] beliefDist = beliefState.toDistributionOverStates(pomdp);
      int nrActions = tree.pomdp.getNumChoicesForObservation(beliefState.so);
      for (int action = 0 ; action < nrActions ; action++) {
        double rewardNow = pomdp.getRewardAfterChoice(beliefState, action, mdpRewards);
        tree.QaUpper.get(beliefIndex).set(action, rewardNow);
        HashMap<Integer, Double> possibleObservations = tree.pomdp.computeObservationProbsAfterAction(beliefDist, action);
        
        int obsNr = 0;
        for (Map.Entry<Integer, Double> entry : possibleObservations.entrySet()) {
          Double obsProbability = entry.getValue();
          int childIndex = tree.beliefAndActionChildren.get(beliefIndex).get(action).get(obsNr);
          double childUpperValue = tree.VUpper.get(childIndex);
          
          tree.QaUpper.get(beliefIndex).set(action, tree.QaUpper.get(beliefIndex).get(action) + obsProbability * childUpperValue);
          obsNr++;
        }
        
        if (Math.abs(tree.QaUpper.get(beliefIndex).get(action)) < 1e-10) { // help with precision
          tree.QaUpper.get(beliefIndex).set(action, 0.0);
        }
        
      }
      tree.VUpper.set(beliefIndex, Collections.max(tree.QaUpper.get(beliefIndex)));
    }

    private Belief stateToBeliefState(int s) {
      double[] beliefState = new double[pomdp.getNumStates()];
      Arrays.fill(beliefState, 0.0);
      beliefState[s] = 1.0;
      return new Belief(beliefState, pomdp);
    }

    private Double[] getMaxAlphaVal(Belief childBelief) {
      Double maxVal = Double.NEGATIVE_INFINITY;
      double[] childBeliefDist = childBelief.toDistributionOverStates(pomdp);
      Double[] maxAlpha = new Double[pomdp.getNumStates()];
      for (Map.Entry<Object,Double[]> alpha : tree.VsLowerBound) {
          Double value = HSVITree.dot(alpha.getValue(), childBeliefDist);
          if (value > maxVal) {
            maxVal = value;
            maxAlpha = alpha.getValue().clone();
          }
      }
      return maxAlpha;
    }

    public double getPruneThreshold() {
      return pruneThreshold;
    }
    
    private boolean shouldPruneAlphas() {
      return (1.0 * (tree.VsLowerBound.size() - tree.pruneData.lastGammaSize) / tree.pruneData.lastGammaSize)
          > tree.pruneData.pruneThreshold; 
    }
    
    private void prune() {
      if (shouldPruneAlphas())
        pruneAlpha();
    }
    
    public void pruneAlpha() {
      List<Integer> bValid = new ArrayList<>();
      for (int i = 0; i < tree.beliefNodes.size(); i++) {
          if (!tree.bPruned.get(i)) {
              bValid.add(i);
          }
      }
      
      BitSet pruned = new BitSet();

      for (int i = 0; i < tree.VsLowerBound.size(); i++) {
          if (pruned.get(i)) 
            continue;
          for (int j = Math.max(tree.VsLowerBoundInitial.size(), i + 1); j < tree.VsLowerBound.size(); j++) {
              if (pruned.get(j)) 
                continue;
              boolean[] dominance = beliefSpaceDomination(tree.VsLowerBound.get(i).getValue(),
                                                          tree.VsLowerBound.get(j).getValue(),
                                                          bValid);
              if (dominance[0]) {
                  pruned.set(j);
              } else if (dominance[1]) {
                  pruned.set(i);
                  break;
              }
          }
      }

      // Remove pruned elements
      for (int i = tree.VsLowerBound.size() - 1; i >= 0; i--) {
          if (pruned.get(i)) {
              tree.VsLowerBound.remove(i);
          }
      }
      tree.pruneData.lastGammaSize = tree.VsLowerBound.size();
  }
    
    public boolean[] beliefSpaceDomination(Double[] alpha1, Double[] alpha2, List<Integer> bValid) {
      boolean a1Dominant = true;
      boolean a2Dominant = true;

      for (Integer beliefIndex: bValid) {
          if (!a1Dominant && !a2Dominant) {
              return new boolean[]{false, false};
          }
          double deltaV = intersectionDistance(alpha1, alpha2, beliefIndex);
          if (deltaV <= delta) {
              a1Dominant = false;
          }
          if (deltaV >= -delta) {
              a2Dominant = false;
          }
      }
      return new boolean[]{a1Dominant, a2Dominant};
  }

  public double intersectionDistance(Double[] alpha1, Double[] alpha2, Integer beliefIndex) {
      double s = 0.0;
      double dotSum = 0.0;
      double[] beliefNode = tree.beliefNodes.get(beliefIndex).toDistributionOverStates(pomdp);
      

      for (int i = 0; i < alpha1.length; i++) {
          double diff = alpha1[i] - alpha2[i];
          s += (diff * diff);
          dotSum += diff * beliefNode[i];
      }
      
      return dotSum / Math.sqrt(s);
  }
    
}


