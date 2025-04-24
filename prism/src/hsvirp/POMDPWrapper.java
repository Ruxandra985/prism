package hsvirp;

import java.util.AbstractMap;
import java.util.ArrayList;
import java.util.BitSet;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map.Entry;
import explicit.Belief;
import explicit.POMDP;
import prism.PrismUtils;

public class POMDPWrapper {
  // this will hold the POMDP and the target nodes and will 
  // override some transitions and rewards functions
  
  private POMDP<Double> pomdp;
  private BitSet target;
  
  private int sinkState;
  private Object sinkAction;
  private int sinkObservation;
  private int sinkUnobservation;
  
  public POMDPWrapper(POMDP<Double> pomdp, BitSet target){
    this.pomdp = pomdp;
    this.target = target;
    this.sinkState = pomdp.getNumStates(); // because indexed from 0
    this.sinkAction = pomdp.getNumChoices();
    this.sinkObservation = pomdp.getNumObservations();
    this.sinkUnobservation = pomdp.getNumUnobservations();
  }

  public POMDP<Double> getPomdp() {
    return pomdp;
  }

  public BitSet getTarget() {
    return target;
  }
  
  public Integer getSinkState() {
    return sinkState;
  }

  public double[] beliefToDistributionOverStates(Belief beliefRealNode) {
    // if the belief has observation sinkObservation then we are in sinkState clearly
    double[] distributionOverStates = new double[pomdp.getNumStates() + 1];
    int n = pomdp.getNumStates();
    for (int s = 0; s < n; s++) {
      if (pomdp.getObservation(s) == beliefRealNode.so) {
        int unobserv = pomdp.getUnobservation(s);
        distributionOverStates[s] = beliefRealNode.bu[unobserv];
      }
    }
    if (beliefRealNode.so == sinkObservation) {
      // we are 100% in sink state
      distributionOverStates[sinkState] = 1.0;
    }
    PrismUtils.normalise(distributionOverStates);
    return distributionOverStates;
  }

  public HashMap<Integer, Double> computeObservationProbsAfterAction(double[] beliefDist, int choiceNr) {
    // okay first consider the observation probs after action for beliefDist
    // ignoring target and sinkState
    
    double[] beliefDistSimple = new double[pomdp.getNumStates()];
    double sum = 0.0;
    
    for (int i = 0 ; i < pomdp.getNumStates(); i++) {
      if (i != sinkState && !target.get(i)) {
        beliefDistSimple[i] = beliefDist[i];
        sum += beliefDist[i];
      }
    }
    // note that beliefDistSimple is not normalized and does not have to be
    // for all the states that do not lead to sinkState, compute the usual way
    HashMap<Integer, Double> hmap = pomdp.computeObservationProbsAfterAction(beliefDistSimple, choiceNr);
    
    // with probability 1.0 - sum we are in a target / sink state, in which case 
    // choiceNr leads to us observing sinkObservation with probability 1.0
    
    if (sum != 1.0)
      hmap.put(sinkObservation, 1.0-sum);
    return hmap;
    
  }

  public double getRewardAfterChoice(Belief beliefState, int choiceNr) {
    // we get a reward when we enter a target state
    double[] beliefDist = beliefToDistributionOverStates(beliefState);
    
    double reward = 0.0;
    
    for (int s = 0 ; s < beliefDist.length; s++) {
      if (beliefDist[s] > 0.0)
        reward += beliefDist[s] * getTransitionReward(s, choiceNr); 
    }
    
    return reward;
  }

  public int getNumChoicesForObservation(int so) {
    // when the observation is sinkObservation, the belief state can ONLY
    // be sinkState
    if (so != sinkObservation)
      return pomdp.getNumChoicesForObservation(so);
    
    // given we are clearly in sinkState, we shall just say we have one choice
    // we don't really care to allow multiple actions
    return 1;
  }

  public Belief getBeliefAfterChoiceAndObservation(Belief beliefState, int choiceNr, Integer observation) {
    // this should work as expected if we cannot be in sink / target
    // if in sink or target, you can only have choice nr 0 and observation sinkObservation
    double[] beliefInDist = beliefToDistributionOverStates(beliefState);
    double[] nextBeliefInDist = getBeliefInDistAfterChoiceAndObservation(beliefInDist, choiceNr, observation);
    
    return stateDistToBelief(nextBeliefInDist);
  }

  public double[] getBeliefInDistAfterChoiceAndObservation(double[] beliefInDist, int choiceNr, Integer observation) {
    int n = beliefInDist.length;
    double[] nextBelief = new double[n];
    double[] beliefAfterAction = getBeliefInDistAfterChoice(beliefInDist, choiceNr);
    double prob;
    for (int s = 0; s < n; s++) {
      prob = beliefAfterAction[s] * getObservationProb(s, observation);
      nextBelief[s] = prob;
    }
    PrismUtils.normalise(nextBelief);
    return nextBelief;
  }

  public double[] getBeliefInDistAfterChoice(double[] beliefInDist, int choiceNr) {
    int n = beliefInDist.length;
    double[] nextBeliefInDist = new double[n];
    for (int sp = 0; sp < n; sp++) {
      if (beliefInDist[sp] >= 1.0e-6 && !target.get(sp) && sp != sinkState) {
        Iterator<Entry<Integer, Double>> tIter = pomdp.getTransitionsIterator(sp, choiceNr);
        while (tIter.hasNext()) {
          Entry<Integer, Double> e = tIter.next();
          int s = e.getKey();
          double prob = e.getValue();
          nextBeliefInDist[s] += beliefInDist[sp] * prob;
        }
      }
      else if (beliefInDist[sp] >= 1.0e-6 && (target.get(sp) || sp == sinkState)) {
        // if in sink state or target, then choice will lead to sink
        nextBeliefInDist[sinkState] += beliefInDist[sp];
      }
    }
    return nextBeliefInDist;
  }

  public Belief getBeliefAfterChoice(Belief beliefState, int choiceNr) {
    double[] beliefInDist = beliefToDistributionOverStates(beliefState);
    double[] nextBeliefInDist = getBeliefInDistAfterChoice(beliefInDist, choiceNr);
    return stateDistToBelief(nextBeliefInDist);
  }

  public int getNumStates() {
    return pomdp.getNumStates() + 1;
  }

  public Belief getInitialBelief() {
    double[] initialDist = new double[pomdp.getNumStates() + 1];
    int nrInitStates = pomdp.getNumInitialStates(); // should be = 1
    Iterator<Integer> initIter = pomdp.getInitialStates().iterator();
    
    while (initIter.hasNext()) {
      Integer initState = initIter.next();
      initialDist[initState] = 1.0 / nrInitStates; // assume eq prob for each init state
    }
    return stateDistToBelief(initialDist);
  }

  public int getNumObservations() {
    return pomdp.getNumObservations() + 1;
  }

  public Object getActionForObservation(int so, int action) {
    if (so != sinkObservation)
      return pomdp.getActionForObservation(so, action);
    // unless in sink state we have the normal actions 
    // if in sink state, since we have the special observation we can also have a special act
    return sinkAction;
  }

  public Iterator<Entry<Integer, Double>> getTransitionsIterator(int s, int choiceNrFromS) {
    if (s != sinkState && !target.get(s))
      return pomdp.getTransitionsIterator(s, choiceNrFromS);
    
    // if target or sink, we just go to sinkState with every action
    ArrayList<Entry<Integer, Double>> transitions = new ArrayList<>();
    transitions.add(new AbstractMap.SimpleEntry<>(sinkState, 1.0));
    return transitions.iterator();
  }

  public int getChoiceByAction(int s, Object actionLabel) {
    // if not in sink state, we don't particularly care
    if (s != sinkState)
      return pomdp.getChoiceByAction(s, actionLabel);
    
    // if in sink state, actionLabel has to be sinkAction
    
    if (actionLabel != sinkAction)
      return -1;
    
    return 0;
  }

  public double getObservationProb(int stateInner, int obs) {
    if (stateInner != sinkState) // only sinkState has sinkObservation
      return pomdp.getObservationProb(stateInner, obs);
    else if (obs == sinkObservation)
      return 1.0;
    else return 0.0;
  }

  public int getObservation(int stateInner) {
    if (stateInner != sinkState)
      return pomdp.getObservation(stateInner);
    return sinkObservation;
  }

  public Belief stateDistToBelief(double[] beliefState) {
    int so = -1;
    double[] bu = new double[pomdp.getNumUnobservations() + 1];
    for (int s = 0; s < beliefState.length; s++) {
      if (beliefState[s] != 0) {
        if (s != sinkState) {
          so = pomdp.getObservation(s);
          bu[pomdp.getUnobservation(s)] += beliefState[s];
        }
        else {
          so = sinkObservation;
          bu[sinkUnobservation] += beliefState[s];
          // this should not mean anything other than enforce a proper distribution over
          // unobservations
        }
      }
    }
    return new Belief(so, bu);
  }

  public int getNumChoices(int state) {
    // if state is not sink, it has as many choices as usual
    if (state != sinkState)
      return pomdp.getNumChoices(state);
    // sink state has only one choice
    return 1;
  }

  public Object getAction(int state, int i) {
    // if not in sink state, we don't care
    if (state != sinkState)
      return pomdp.getAction(state, i);
    // if sinkState, i must be 0
    if (i == 0)
      return sinkAction;
    // this should not happen
    return null;
  }

  public double mvMultSingle(int state, int i, double[] myActionVector) {
    if (state != sinkState && !target.get(state))
      return pomdp.mvMultSingle(state, i, myActionVector);
    else {
      // you have only one successor, sinkState with probability 1.0 regardless of i
      return myActionVector[sinkState];
    }
  }

  public Double getTransitionReward(int state, int i) {
    // if you have probability p of getting into a target node, the reward should be p
    Iterator<Entry<Integer, Double>> tIter = getTransitionsIterator(state, i);
    Double reward = 0.0;
    
    while (tIter.hasNext()) {
      Entry<Integer, Double> e = tIter.next();
      Integer nextState = e.getKey();
      Double prob = e.getValue();
      if (target.get(nextState))
        reward += prob;
    }
    
    return reward;
    
  }
  
  public static boolean isTerminal(POMDPWrapper pomdpWrapper, Belief currNode) {
    double[] nodesDist = pomdpWrapper.beliefToDistributionOverStates(currNode);
    for (int i = 0; i < nodesDist.length; i++) {
      if (nodesDist[i] > 0.0 && !pomdpWrapper.target.get(i) && i != pomdpWrapper.getSinkState()) {
          return false; // we could be in a non-target state, then this is not a target belief state
      }
    }
    return true;
  }
  
  public static boolean isTerminalState(POMDPWrapper pomdpWrapper, int state) {
    return (pomdpWrapper.target.get(state) || state == pomdpWrapper.getSinkState());
  }

  
  
  
  
}
