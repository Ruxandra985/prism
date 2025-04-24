package explicit;

import prism.ModelType;

public interface POMDPHSVIRP<Value> extends POMDP<Value> {
  @Override
  default ModelType getModelType()
  {
    return ModelType.POMDPHSVIRP;
  }
}
