package hsvirp;

import java.util.*;

import explicit.Belief;
import explicit.POMDP;
import explicit.rewards.MDPRewards;

class PruneData {
    int lastGammaSize;
    int lastBSize;
    double pruneThreshold;

    public PruneData(int lastGammaSize, int lastBSize, double pruneThreshold) {
        this.lastGammaSize = lastGammaSize;
        this.lastBSize = lastBSize;
        this.pruneThreshold = pruneThreshold;
    }
}

class HSVITree {
    POMDP<Double> pomdp;
    MDPRewards<Double> mdpRewards;
    BitSet target;
    BitSet remain;

    List<Belief> beliefNodes; // Belief vectors
    List<Boolean> beliefNodeHasChildren; // Belief-action children
    List<List<List<Integer>>> beliefAndActionChildren; // Belief-action-observation children
    //beliefAndActionChildren[i][j][k] = index of child of beliefNodes[i] taking choice j and obs k
    List<Double> VsUpperBound;
    
    
    List<Map.Entry<Object, Double[]>> VsLowerBound;
    List<Map.Entry<Object, Double[]>> VsLowerBoundInitial;
    List<Double> VUpper;
    // VUpper[i] = VU(beliefNodes[i]) 
    List<Double> VLower;
    // similar as VUpper
    
    List<List<Double>> QaLower;
    List<List<Double>> QaUpper;

    List<Integer> frontier; // The current frontier for traversal
    List<Integer> real;
    
    BitSet targetNodes; // which states are considered terminal
    
    BitSet bPruned;
    List<BitSet> baPruned; 
    PruneData pruneData;
    
    List<Integer> beliefSampleFrequency; // # times sampled belief state b
    List<List<Integer>> beliefActionSampleFrequency; // # times sampled belief state b and action a from it
    List<List<List<Integer>>> beliefActObsSampleFrequency; // # times sampled belief b, action a and observation o
    

    public HSVITree(POMDP<Double> pomdp, HSVIRPSolver solver, BitSet target, BitSet remain, MDPRewards<Double> mdpRewards) {
        this.pomdp = pomdp;
        this.target = target;
        this.remain = remain;
        this.mdpRewards = mdpRewards;
        
        Set<Map.Entry<Object, Double[]>> upperBounds = solver.computeUpperBoundPolicy(pomdp, mdpRewards, target, remain);
        List<Double> cornerValues = new ArrayList<>();
        
        for (int i = 0 ; i < pomdp.getNumStates() ; i++)
          cornerValues.add(Double.NEGATIVE_INFINITY);
        
        for (Map.Entry<Object,Double[]> alpha : upperBounds) {
          Double[] alphaVector = alpha.getValue();
          for (int i = 0 ; i < pomdp.getNumStates() ; i++)
            cornerValues.set(i, Math.max(cornerValues.get(i), alphaVector[i]));
        }
        this.VsUpperBound = cornerValues;
        
        Set<Map.Entry<Object, Double[]>> alphaVectorsLower = solver.computeLowerBoundPolicy(pomdp, mdpRewards, remain);
        this.VsLowerBound = new ArrayList<>(alphaVectorsLower);
        this.VsLowerBoundInitial = new ArrayList<>(alphaVectorsLower);
        

        this.beliefNodes= new ArrayList<>();
        this.beliefNodeHasChildren = new ArrayList<>();
        this.VUpper = new ArrayList<>();
        this.VLower = new ArrayList<>();
        this.QaLower = new ArrayList<>();
        this.QaUpper = new ArrayList<>();

        this.beliefAndActionChildren = new ArrayList<>();
        this.frontier = new ArrayList<>();
        this.real = new ArrayList<>();
        
        this.targetNodes = target;
        
        this.bPruned = new BitSet();
        this.baPruned = new ArrayList<>();
        this.pruneData = new PruneData(this.VsLowerBoundInitial.size(), 0, solver.getPruneThreshold());
        
        this.beliefSampleFrequency = new ArrayList<>();
        this.beliefActionSampleFrequency = new ArrayList<>();
        this.beliefActObsSampleFrequency = new ArrayList<>();
    }

    public static HSVITree initializeTree(HSVIRPSolver solver, POMDP<Double> pomdp, BitSet target, BitSet remain, MDPRewards<Double> mdpRewards) {
        HSVITree tree = new HSVITree(pomdp, solver, target, remain, mdpRewards);

        Belief initialBelief = processState(tree, pomdp.getInitialBelief());
        insertRoot(pomdp, tree, initialBelief);

        return tree;
    }

    private static Belief processState(HSVITree tree, Belief initialBelief) {
      // normalise to get only states you want to remain in
      double[] beliefDist = initialBelief.toDistributionOverStates(tree.pomdp);
      
      for (int i = 0 ; i < tree.pomdp.getNumStates() ; i++) {
        if (tree.remain != null && !tree.remain.get(i)) {
          beliefDist[i] = 0.0;
        }
      }
      
      return initialBelief;
    }

    private static void insertRoot(POMDP<Double> pomdp, HSVITree tree, Belief belief) {
        
        Double[] vsUpperBoundArray = new Double[pomdp.getNumStates()];
        tree.VsUpperBound.toArray(vsUpperBoundArray); // fill the array

        tree.beliefNodes.add(belief);
        tree.beliefNodeHasChildren.add(false);
        tree.beliefAndActionChildren.add(new ArrayList<>());
        tree.VUpper.add(dot(vsUpperBoundArray, belief.toDistributionOverStates(pomdp)));
        tree.VLower.add(lowerValue(tree, belief));
        tree.QaLower.add(new ArrayList<>());
        tree.QaUpper.add(new ArrayList<>());
        
        tree.real.add(tree.beliefNodes.indexOf(belief)); // the index of the root is 0, wrote this for clarity
        
        tree.bPruned.set(tree.beliefNodes.indexOf(belief), false);
        tree.baPruned.add(new BitSet());
        
        tree.beliefSampleFrequency.add(0);
        tree.beliefActionSampleFrequency.add(new ArrayList<>());
        tree.beliefActObsSampleFrequency.add(new ArrayList<>());
        fillBelief(tree, tree.beliefNodes.indexOf(belief)); // again, index of root is actually 0
    }

    public static Double[] update (HSVITree tree, int beliefIndex, int action, int observation) {
      
      int innerState = tree.beliefAndActionChildren.get(beliefIndex).get(action).get(observation);
      Belief beliefNextNode = tree.beliefNodes.get(innerState);
      Double vUp = 0.0;
      Double vLow = 0.0;
      if (!HSVITree.isTerminal(tree, beliefNextNode)) {
        vLow = lowerValue(tree, beliefNextNode);
        vUp = upperValue(tree, beliefNextNode);
      }
      
      tree.VLower.set(innerState, vLow);
      tree.VUpper.set(innerState, vUp);
      
      return new Double[]{(double) innerState, vLow, vUp};
    }

    public static Double[] addBelief(HSVITree tree, Belief newBelief) {
      
      //if (newBelief.so == 3) {
        //System.out.println("good state");
      //}
      
      newBelief = processState(tree, newBelief);
      
      for (int i = 0 ; i < tree.beliefNodes.size() ; i++) {
        
        if (tree.beliefNodes.get(i).so == newBelief.so) {
        
          boolean equal = true;
          for (int j = 0 ; j < tree.beliefNodes.get(i).bu.length; j++) {
            if (Math.abs(tree.beliefNodes.get(i).bu[j] - newBelief.bu[j]) > 1e-10)
              equal = false;
          }
          if (equal) {
            return new Double[] {(double)i, tree.VLower.get(i), tree.VUpper.get(i)};
          }
          
        }
      }
      
      tree.beliefNodes.add(newBelief);
      int newBeliefIndex = tree.beliefNodes.size() - 1;
      tree.beliefNodeHasChildren.add(false);
      tree.beliefAndActionChildren.add(new ArrayList<>());
      
      tree.beliefSampleFrequency.add(0);
      tree.beliefActionSampleFrequency.add(new ArrayList<>());
      tree.beliefActObsSampleFrequency.add(new ArrayList<>());
      
      tree.QaLower.add(new ArrayList<>());
      tree.QaUpper.add(new ArrayList<>());
      
      Double vUp = 0.0;
      Double vLow = 0.0;
      if (!HSVITree.isTerminal(tree, newBelief)) {
        vLow = lowerValue(tree, newBelief);
        vUp = upperValue(tree, newBelief);
      }
      
      tree.VLower.add(vLow);
      tree.VUpper.add(vUp);
      tree.bPruned.set(newBeliefIndex, false);
      tree.baPruned.add(new BitSet());
      
      return new Double[]{(double) newBeliefIndex, vLow, vUp};
    }

    public static void fillBelief(HSVITree tree, int beliefIndex) {
        if (!tree.beliefNodeHasChildren.get(beliefIndex)) { // we did not store any children yet
            fillUnpopulated(tree, beliefIndex);
        } else { // the given belief state already has some nr of children
            fillPopulated(tree, beliefIndex);
        }
    }
    
    

    private static void fillUnpopulated(HSVITree tree, int beliefIndex) {
      Belief beliefState = tree.beliefNodes.get(beliefIndex);
      int nrActions = tree.pomdp.getNumChoicesForObservation(beliefState.so);
      
      tree.QaLower.set(beliefIndex, new ArrayList<>());
      tree.QaUpper.set(beliefIndex, new ArrayList<>());
      
      for (int choiceNr = 0 ; choiceNr < nrActions; choiceNr++) {
        addAction(beliefIndex, choiceNr, tree);
        // where in beliefAndActionChildren we will
        
        double beliefReward = tree.pomdp.getRewardAfterChoice(beliefState, choiceNr, tree.mdpRewards);
        
        double qUpper = beliefReward;
        double qLower = beliefReward;
        
        double[] beliefDist = beliefState.toDistributionOverStates(tree.pomdp);
        
        HashMap<Integer, Double> observationsInfo = tree.pomdp.computeObservationProbsAfterAction(beliefDist, choiceNr);
         
        tree.beliefActObsSampleFrequency.get(beliefIndex).add(new ArrayList<>());
        
        for (Map.Entry<Integer, Double> entry : observationsInfo.entrySet()) {
          
          Integer observation = entry.getKey();
          Double observationProbability = entry.getValue();
          
          Belief predictedBeliefObs = tree.pomdp
              .getBeliefAfterChoiceAndObservation(beliefState, choiceNr, observation);
            
          // it only makes sense to add the child belief if you can reach it
          Double[] results = addBelief(tree, predictedBeliefObs);
          int nextBeliefIndex = results[0].intValue();
          double nextVLower = results[1];
          double nextVUpper = results[2];
            
          qUpper += observationProbability * nextVUpper;
          qLower += observationProbability * nextVLower;
          
          tree.beliefAndActionChildren.get(beliefIndex)
            .get(choiceNr)
            .add(nextBeliefIndex);
          
          tree.beliefActObsSampleFrequency.get(beliefIndex).get(choiceNr).add(0);
          
        }
        
        tree.beliefNodeHasChildren.set(beliefIndex, true);
        tree.QaLower.get(beliefIndex).add(qLower);
        tree.QaUpper.get(beliefIndex).add(qUpper);// inner lists indexed in order of choice nr
        tree.beliefActionSampleFrequency.get(beliefIndex).add(0); // initialize or current action 
        
      }
      
      tree.VLower.set(beliefIndex, lowerValue(tree, beliefState));
      
      tree.VUpper.set(beliefIndex, Collections.max(tree.QaUpper.get(beliefIndex)));
      
    }
    
    private static void fillPopulated(HSVITree tree, int beliefIndex) {
      Belief beliefState = tree.beliefNodes.get(beliefIndex);
      
      int nrActions = tree.pomdp.getNumChoicesForObservation(beliefState.so);
      
      for (int choiceNr = 0 ; choiceNr < nrActions; choiceNr++) {
        // skip if this mix of action and state is pruned
        
        if (tree.baPruned.get(beliefIndex).get(choiceNr))
          continue;
        
        double beliefReward = tree.pomdp.getRewardAfterChoice(beliefState, choiceNr, tree.mdpRewards);
        
        double qUpper = beliefReward;
        double qLower = beliefReward;
        
        double[] beliefDist = beliefState.toDistributionOverStates(tree.pomdp);
        
        HashMap<Integer, Double> observationsInfo = tree.pomdp.computeObservationProbsAfterAction(beliefDist, choiceNr);
        
        int obsNr = 0;
        
        for (Map.Entry<Integer, Double> entry : observationsInfo.entrySet()) {
          
          Double observationProbability = entry.getValue();
          
          Double[] results = update(tree, beliefIndex, choiceNr, obsNr);
          
          double nextVLower = results[1];
          double nextVUpper = results[2];
          
          qUpper += observationProbability * nextVUpper;
          qLower += observationProbability * nextVLower;
          
          obsNr++;
          
        }
        
        tree.QaUpper.get(beliefIndex).set(choiceNr, qUpper);
        tree.QaLower.get(beliefIndex).set(choiceNr, qLower);
      }
      
      tree.VLower.set(beliefIndex, lowerValue(tree, beliefState));
      tree.VUpper.set(beliefIndex, Collections.max(tree.QaUpper.get(beliefIndex)));
    }

    private static void addAction(int beliefIndex, int choiceNr, HSVITree tree) {
      tree.beliefAndActionChildren.get(beliefIndex).add(new ArrayList<>(tree.pomdp.getNumObservations()));
      tree.baPruned.get(beliefIndex).set(choiceNr, true);
    }

    

    static double dot(Double[] vec1, double[] vec2) {
        double sum = 0.0;
        for (int i = 0; i < vec1.length; i++) {
            sum = Math.fma(vec1[i], vec2[i], sum);
        }
        return sum;
    }
    
    private static Double lowerValue(HSVITree tree, Belief belief) {
      Double maxVal = Double.NEGATIVE_INFINITY; 

      // Iterate over each alpha vector
      for (Map.Entry<Object, Double[]> alphaVector: tree.VsLowerBound) {
        Double[] vect = alphaVector.getValue();
        // Compute the dot product of vect and the belief state
        Double newVal = dot(vect, belief.toDistributionOverStates(tree.pomdp));

        // Update MAX_VAL if the new value is greater
        maxVal = Math.max(maxVal, newVal);
      }

      return maxVal;
    }
    
    private static Double upperValue(HSVITree tree, Belief belief) {
      // Retrieve corner alpha vectors and compute VCorner
      Double[] cornerAlphas = new Double[tree.VsUpperBound.size()];
      tree.VsUpperBound.toArray(cornerAlphas);
      double[] beliefDist = belief.toDistributionOverStates(tree.pomdp);
      double vCorner = dot(cornerAlphas, beliefDist);

      double vMin = Double.POSITIVE_INFINITY;

      // Loop through each belief index in tree.real
      for (int beliefIndex : tree.real) {
          Belief beliefRealNode = tree.beliefNodes.get(beliefIndex);
        
          // Skip if the belief is terminal
          if (HSVITree.isTerminal(tree, beliefRealNode)) {
              continue;
          }

          double vUpperNode = tree.VUpper.get(beliefIndex);
          double[] beliefRealDist = beliefRealNode.toDistributionOverStates(tree.pomdp);
          double phi = Double.POSITIVE_INFINITY;
          
          for (int i = 0 ; i < beliefDist.length ; i++) {
            if (beliefDist[i] != 0.0 && beliefRealDist[i] != 0.0) {
              if (phi > beliefDist[i] / beliefRealDist[i]) 
                phi = beliefDist[i] / beliefRealDist[i];
            }
            else if (beliefDist[i] == 0.0 && beliefRealDist[i] != 0.0) {
              phi = 0.0;
              break;
            }
          }

          // Compute v
          double v = vCorner + phi * (vUpperNode - dot(cornerAlphas, beliefRealDist));

          // Update vMin
          vMin = Math.min(v, vMin);
      }

      return vMin;
    }
    
    public static boolean isTerminal(HSVITree tree, Belief currNode) {
      double[] nodesDist = currNode.toDistributionOverStates(tree.pomdp);
      for (int i = 0; i < nodesDist.length; i++) {
        if (nodesDist[i] > 1e-10 && !tree.target.get(i)) {
            return false; // we could be in a non-target state, then this is not a target belief state
        }
      }
      return true;
    }
    
    public static boolean isTerminalState(HSVITree tree, int state) {
      return tree.target.get(state);
    }
}

