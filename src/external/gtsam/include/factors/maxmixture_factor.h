/**
 * Copyright 2019 Massachusetts Institute of Technology
 *
 * @file maxmixture_factor.h
 * @author Kevin Doherty
 */

#pragma once

#include <vector>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/nonlinear/Symbol.h>
#include <math.h>
#include <algorithm>

namespace maxmixture {

  /**
   * @brief GTSAM implementation of a max-mixture factor
   *
   * r(x) = min_i -log(w_i) + r_i(x)
   *
   * The error returned from this factor is the minimum error + weight
   * over all of the component factors
   * See Olson and Agarwal RSS 2012 for details
   */
  template <class T>
  class MaxMixtureFactor : public gtsam::NonlinearFactor {

  private:
    std::vector<T> factors_;
    std::vector<double> log_weights_;

  public:
    using Base = gtsam::NonlinearFactor;

    MaxMixtureFactor() = default;

    explicit MaxMixtureFactor(const std::vector<gtsam::Key> keys,
                              const std::vector<T> factors,
                     const std::vector<double> weights) :
    Base(keys) {
      factors_ = factors;
      for (int i = 0; i < weights.size(); i++) {
        log_weights_.push_back(log(weights[i]));
      }
    }

    MaxMixtureFactor& operator=(const MaxMixtureFactor& rhs) {
      Base::operator=(rhs);
      this->factors_ = rhs.factors_;
      this->log_weights_ = rhs.log_weights_;
    }

    virtual ~MaxMixtureFactor() = default;

    double error(const gtsam::Values& values) const override {
      double min_error = std::numeric_limits<double>::infinity();
      for (int i = 0; i < factors_.size(); i++) {
        double error = factors_[i].error(values) - log_weights_[i];
        if (error < min_error) {
          min_error = error;
        }
      }
      return min_error;
    }

    size_t dim() const override {
      if (factors_.size() > 0) {
        return factors_[0].dim();
      }
      else {
        return 0;
      }
    }

    boost::shared_ptr<gtsam::GaussianFactor> linearize(const gtsam::Values& x) const override {
      double min_error = std::numeric_limits<double>::infinity();
      int idx_min = -1;
      for (int i = 0; i < factors_.size(); i++) {
        double error = factors_[i].error(x) - log_weights_[i];
        if (error < min_error) {
          min_error = error;
          idx_min = i;
        }
      }
      return factors_[idx_min].linearize(x);
    }

    int computeMaxIndex(const gtsam::Values& values){
      int max_component_index = -1;
      double min_error = std::numeric_limits<double>::infinity();
      for (int i = 0; i < factors_.size(); i++) {
        double error = factors_[i].error(values);
        //std::cout<<"the error on node "<<i<<" is "<< error<<std::endl;
        if (error < min_error) {
          min_error = error;
          max_component_index = i;
        }
      }
      return max_component_index;
    }
    //temporary fix
    boost::shared_ptr<T> get_factor(int i){
      return boost::dynamic_pointer_cast<T>(factors_[i].clone());
    }
    /* size_t dim() const override { return 1; } */


  };

} // Namespace maxmixture
