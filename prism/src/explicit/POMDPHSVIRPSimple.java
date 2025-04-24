package explicit;

public class POMDPHSVIRPSimple<Value> extends POMDPSimple<Value> implements POMDPHSVIRP<Value> {
  // inherits all methods of POMDPSimple but uses ModelType as declared in POMDPHSVIRP
  //Constructors
  
   /**
    * Constructor: empty POMDP.
    */
   public POMDPHSVIRPSimple()
   {
     super();
   }
  
   /**
    * Constructor: new POMDP with fixed number of states.
    */
   public POMDPHSVIRPSimple(int numStates)
   {
     super(numStates);
   }
  
   /**
    * Copy constructor.
    */
   public POMDPHSVIRPSimple(POMDPSimple<Value> pomdp)
   {
     super(pomdp);
   }
  
   /**
    * Construct a POMDP from an existing one and a state index permutation,
    * i.e. in which state index i becomes index permut[i].
    */
   public POMDPHSVIRPSimple(POMDPHSVIRPSimple<Value> pomdp, int permut[])
   {
     super(pomdp, permut);
   }
  
   /**
    * Construct a POMDP from an existing MDP.
    */
   public POMDPHSVIRPSimple(MDPSimple<Value> mdp)
   {
     super(mdp);
   }
}
