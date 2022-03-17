/**
 * Copyright 2021 Massachusetts Institute of Technology
 * @file R2RANGE example
 * @author: Qiangqiang Huang
 */

#include <gtsam/geometry/Pose2.h>

#include <gtsam/inference/Key.h>
#include <gtsam/inference/Symbol.h>

#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>

#include <gtsam/nonlinear/NonlinearFactorGraph.h>

#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/Marginals.h>
#include <math.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <random>
#include <iostream>
#include <fstream>
#include <string.h>
#include <chrono>
#include "factors/maxmixture_factor.h"
#include <bits/stdc++.h>
#include <sys/stat.h>
#include <sys/types.h>

#define PI 3.14159265

using namespace std;
using namespace gtsam;
std::default_random_engine generator;
std::uniform_real_distribution<double> distribution(0.0,1.0);
bool true_fcn(Key key){
  return true;
}

class RangeFactor_2D: public NoiseModelFactor2<Point2, Point2> {
  double mrange_;

public:
  // The constructor requires the variable key, the (X, Y) measurement value,
  // and the noise model.
  RangeFactor_2D(Key i,Key j, double range, const SharedNoiseModel& model):
    NoiseModelFactor2<Point2, Point2>(model,i, j), mrange_(range){};

  virtual ~RangeFactor_2D() {}

  Vector evaluateError(const Point2& p,const Point2& q,
                       boost::optional<Matrix&> H1 = boost::none,
                       boost::optional<Matrix&> H2 = boost::none) const {
    float tmp_range = sqrt( (p.x() - q.x()) * (p.x() - q.x()) + (p.y() - q.y()) * (p.y() - q.y()));
    if (H1) {

      (*H1) = (Matrix(1,2) << (p.x() - q.x())/tmp_range,
                             (p.y() - q.y())/tmp_range).finished();
      (*H2) = (Matrix(1,2) << -(p.x() - q.x())/tmp_range,
                            -(p.y() - q.y())/tmp_range).finished();
    }
    return (Vector(1) << ( tmp_range - mrange_)).finished();
  }
  Point2 sampleOnRing(double x, double y){
  	double rad = distribution(generator)*PI*2;
  	return Point2(x+mrange_*cos(rad), y+mrange_*sin(rad));
  }  
};

class RangeFactor_SE2: public NoiseModelFactor2<Pose2, Point2> {
  double mrange_;

public:
  // The constructor requires the variable key, the (X, Y) measurement value,
  // and the noise model.
  RangeFactor_SE2(Key i,Key j, double range, const SharedNoiseModel& model):
    NoiseModelFactor2<Pose2, Point2>(model,i, j), mrange_(range){};

  virtual ~RangeFactor_SE2() {}

  Vector evaluateError(const Pose2& p,const Point2& q,
                       boost::optional<Matrix&> H1 = boost::none,
                       boost::optional<Matrix&> H2 = boost::none) const {
    float tmp_range = sqrt( (p.x() - q.x()) * (p.x() - q.x()) + (p.y() - q.y()) * (p.y() - q.y()));
    if (H1) {
      (*H1) = (Matrix(1,3) << (p.x() - q.x())/tmp_range,
                             (p.y() - q.y())/tmp_range, 0.0).finished();
      (*H2) = (Matrix(1,2) << -(p.x() - q.x())/tmp_range,
                            -(p.y() - q.y())/tmp_range).finished();
    }
    return (Vector(1) << ( tmp_range - mrange_)).finished();
  }
  Point2 sampleOnRing(double x, double y){
  	double rad = distribution(generator)*PI*2;
  	return Point2(x+mrange_*cos(rad), y+mrange_*sin(rad));
  } 
};


class UnaryFactor: public NoiseModelFactor1<Pose2> {
	double mx_, my_;

public:
  UnaryFactor(Key j, double x, double y, const SharedNoiseModel& model):
    NoiseModelFactor1<Pose2>(model, j), mx_(x), my_(y) {};

  virtual ~UnaryFactor() {}

  Vector evaluateError(const Pose2& q,
                       boost::optional<Matrix&> H = boost::none) const {
    if (H) {
	    (*H) = (Matrix(2,3) << 1.0,0.0,0.0, 0.0,1.0,0.0).finished();
 
    }
    return (Vector(2) << q.x() - mx_, q.y() - my_).finished();
  }

};

void readConfigFile(const string& configDir, 
                    string& input_dir,
                    string& output_dir,
                    int& inc_step,
                    float& artificial_prior_sigma,
                    int& gt_init){
  artificial_prior_sigma = -1.0;//the default value
  ifstream cFile(configDir);
  if (cFile.is_open())
  {
      string line;
      while(getline(cFile, line)){
          stringstream linestream(line);
          string item;
          linestream >> item;
          if (item.compare("factor_graph_path") == 0){
            linestream >> input_dir;
            cout<<"input_dir is "<<input_dir<<endl;
            continue;            
          }
          else if (item.compare("output_dir") == 0){
            linestream >> output_dir;
            cout<<"output_dir is "<<output_dir<<endl;
            continue;                        
          }
          else if (item.compare("incremental_step") == 0){
            linestream >> inc_step;
            cout<<"inc_step is "<<inc_step<<endl;
            continue;                        
          }
          else if (item.compare("artificial_prior_sigma") == 0){
            linestream >> artificial_prior_sigma;
            cout<<"artificial_prior_sigma is "<<artificial_prior_sigma<<endl;
            continue;
          }
          else if (item.compare("gt_init") == 0){
            linestream >> gt_init;
            cout<<"gt_init is "<<gt_init<<endl;
            continue;
          }
      }
      cout<<"Config file loaded."<<endl;  
      cout<<"----------------------------"<<endl;    
  }
  else {
      std::cerr << "Couldn't open config file for reading.\n";
  }
}


RangeFactor_SE2 constructRangeFactor(string n1_str,
						          string n2_str,
						          double m_r,
						          double sigma_r,
						          unsigned char rbt_chr){
	int n1_idx = stoi(n1_str.substr(1,n1_str.size()-1));
	int n2_idx = stoi(n2_str.substr(1,n2_str.size()-1));
	noiseModel::Diagonal::shared_ptr noise = noiseModel::Isotropic::Sigma(1, sigma_r);
	unsigned char n1_chr = (unsigned char)n1_str[0];
	unsigned char n2_chr = (unsigned char)n2_str[0];
	if (n1_chr == rbt_chr){
	  RangeFactor_SE2 r_factor = RangeFactor_SE2(Symbol(n1_chr, n1_idx),Symbol(n2_chr, n2_idx),
	                               m_r, noise);              
	  return r_factor;
	}
	else{
	  RangeFactor_SE2 r_factor = RangeFactor_SE2(Symbol(n2_chr, n2_idx),Symbol(n1_chr, n1_idx),
	                               m_r, noise);
	  return r_factor;                            
	}
}

BetweenFactor<Pose2> constructPoseFactor(string n1_str,
  string n2_str,
  Pose2 pose,
  Eigen::Matrix3d cov_mat){

  noiseModel::Gaussian::shared_ptr noise = 
    noiseModel::Gaussian::Covariance(cov_mat);
  int n1_idx = stoi(n1_str.substr(1,n1_str.size()-1));
  int n2_idx = stoi(n2_str.substr(1,n2_str.size()-1));
  unsigned char n1_chr = (unsigned char)n1_str[0];
  unsigned char n2_chr = (unsigned char)n2_str[0];
  BetweenFactor<Pose2> p_factor(Symbol(n1_chr, n1_idx),Symbol(n2_chr, n2_idx),
                       pose, noise);
  return p_factor;
}


Symbol string2sym(string n1_str){
  int n1_idx = stoi(n1_str.substr(1,n1_str.size()-1));
  unsigned char n1_chr = (unsigned char)n1_str[0];
  return Symbol(n1_chr, n1_idx);
}

void readFactorGraph(const string& graphDir,
                     vector<Symbol>& syms,
                     NonlinearFactorGraph& graph,
                     Values& vals,
                     float artificial_prior_sigma = -1.0){
  ifstream gFile(graphDir);
  if (gFile.is_open())
  {
    unsigned char rbt_chr = 'X';
    unsigned char lm_chr = 'L';
    string line;
    while(getline(gFile, line)){
      stringstream linestream(line);
      string item;
      linestream >> item;
      if (item.compare("Variable") == 0){
        string pose_or_lmk;
        string SE2_or_R2;
        string node_str;
        double dx;
        double dy;
        double dth;
        //skip Pose/Landmark or R2
        linestream >> pose_or_lmk;
        linestream >> SE2_or_R2;
        //now it comes to node name
        //we require name_name has to be something like X1 or L1
        linestream >> node_str;

        if (SE2_or_R2.compare("R2") == 0){
          //values
          linestream >> dx;
          linestream >> dy;
          int n_idx = stoi(node_str.substr(1,node_str.size()-1));
          unsigned char n_chr = (unsigned char)node_str[0];
          if ( n_chr == rbt_chr || n_chr == lm_chr){
              syms.push_back(Symbol(n_chr, n_idx));
              vals.insert(Symbol(n_chr, n_idx), Point2(dx, dy));
              cout<<"Read node "<<node_str
                <<" with ground truth ("
                <<dx<<","<<dy<<")"<<endl;
              if((artificial_prior_sigma>0.1) && (n_chr == lm_chr)){
                graph.push_back(PriorFactor<Point2>(Symbol(n_chr, n_idx), 
                  Point2(dx, dy), 
                  noiseModel::Isotropic::Sigma(2, artificial_prior_sigma)));
                cout<<"Artificial prior factor of "
                    <<node_str<<" at ("
                    <<dx<<","<<dy<<")"
                    <<" and with sigma of "
                    <<artificial_prior_sigma
                    <<endl;
              }
          }
          else{
              cerr << "An invalid node_str "<<node_str<<endl;
          }          
        }
        else if (SE2_or_R2.compare("SE2") == 0){
          //values
          linestream >> dx;
          linestream >> dy;
          linestream >> dth;
          int n_idx = stoi(node_str.substr(1,node_str.size()-1));
          unsigned char n_chr = (unsigned char)node_str[0];
          if ( n_chr == rbt_chr || n_chr == lm_chr){
              syms.push_back(Symbol(n_chr, n_idx));
              vals.insert(Symbol(n_chr, n_idx), Pose2(dx, dy, dth));
              cout<<"Read node "<<node_str
                <<" with ground truth ("
                <<dx<<","<<dy<<","<<dth<<")"<<endl;
          }
          else{
              cerr << "An invalid node_str "<<node_str<<endl;
          }           
        }
        continue;            
      }
      else if (item.compare("Factor") == 0){
        string factorType;
        linestream >> factorType;
        if (factorType.compare("UnaryR2GaussianPriorFactor") == 0){
          string node_str;
          string cov_str;
          double m_x;
          double m_y;
          double cov_xx;
          double cov_xy;
          double cov_yx;
          double cov_yy;
          linestream >> node_str >> m_x >> m_y>>cov_str 
            >> cov_xx >> cov_xy >> cov_yx >> cov_yy;
          assert(cov_str.compare("covariance") == 0);
          int n_idx = stoi(node_str.substr(1,node_str.size()-1));
          unsigned char n_chr = (unsigned char)node_str[0];
          Eigen::Matrix2d cov_mat;
          cov_mat<<cov_xx, cov_xy, cov_yx, cov_yy;
          noiseModel::Gaussian::shared_ptr priorNoise = 
            noiseModel::Gaussian::Covariance(cov_mat);
          if (n_chr == rbt_chr || n_chr == lm_chr){
              graph.push_back(PriorFactor<Point2>(Symbol(n_chr, n_idx), Point2(m_x, m_y), priorNoise));
              cout<<"Read prior factor of "
                  <<node_str<<" at ("
                  <<m_x<<","<<m_y<<")"
                  <<" and cov_xx("
                  <<cov_xx<<")"
                  <<endl;
          }
          else{
              cerr << "An invalid node_str "<<node_str<<endl;
          }
        }
        else if(factorType.compare("R2RelativeGaussianLikelihoodFactor") == 0){
          string n1_str;
          string n2_str;
          string cov_str;
          double m_x;
          double m_y;
          double cov_xx;
          double cov_xy;
          double cov_yx;
          double cov_yy;
          linestream >> n1_str >> n2_str >> m_x >> m_y 
            >> cov_str 
            >> cov_xx >> cov_xy >> cov_yx >> cov_yy;
          Eigen::Matrix2d cov_mat;
          cov_mat<<cov_xx, cov_xy, cov_yx, cov_yy;
          noiseModel::Gaussian::shared_ptr noise = 
            noiseModel::Gaussian::Covariance(cov_mat);
          assert(cov_str.compare("covariance") == 0);
          int n1_idx = stoi(n1_str.substr(1,n1_str.size()-1));
          int n2_idx = stoi(n2_str.substr(1,n2_str.size()-1));
          unsigned char n1_chr = (unsigned char)n1_str[0];
          unsigned char n2_chr = (unsigned char)n2_str[0];
          if ( (n1_chr == rbt_chr || n1_chr == lm_chr)
               &&
               (n2_chr == rbt_chr || n2_chr == lm_chr)){
              graph.push_back(BetweenFactor<Point2>(Symbol(n1_chr, n1_idx),Symbol(n2_chr, n2_idx),
                                                    Point2(m_x, m_y), noise));
              cout<<"Read factor between "<<n1_str<<" and "
                  <<n2_str<<" with ("
                  <<m_x<<","<<m_y<<")"
                  <<" and cov_xx("
                  <<cov_xx<<")"
                  <<endl;
          }
          else{
              cerr << "Invalid node_str "<<n1_str<<", "<<n2_str<<endl;
          }
        }
        else if(factorType.compare("R2RangeGaussianLikelihoodFactor") == 0){
          string n1_str;
          string n2_str;
          double m_r;
          double sigma_r;
          linestream >> n1_str >> n2_str >> m_r
            >> sigma_r  ;
          int n1_idx = stoi(n1_str.substr(1,n1_str.size()-1));
          int n2_idx = stoi(n2_str.substr(1,n2_str.size()-1));
          noiseModel::Diagonal::shared_ptr noise = noiseModel::Isotropic::Sigma(1, sigma_r);
          unsigned char n1_chr = (unsigned char)n1_str[0];
          unsigned char n2_chr = (unsigned char)n2_str[0];
          if ( (n1_chr == rbt_chr || n1_chr == lm_chr)
               &&
               (n2_chr == rbt_chr || n2_chr == lm_chr)){
            graph.push_back(RangeFactor_2D(Symbol(n1_chr, n1_idx),Symbol(n2_chr, n2_idx),
                                           m_r, noise));
            cout<<"Read range factor between "<<n1_str<<" and "
                <<n2_str<<" with range "
                <<m_r<<endl;
          }
          else{
              cerr << "Invalid node_str "<<n1_str<<", "<<n2_str<<endl;
          }
        }
        else if(factorType.compare("UnarySE2ApproximateGaussianPriorFactor") == 0){
          string node_str;
          string cov_str;
          double m_x, m_y, m_th;
          double cov_xx, cov_xy, cov_xth,
                 cov_yx, cov_yy, cov_yth,
                 cov_thx, cov_thy, cov_thth;

          linestream >> node_str >> m_x >> m_y>> m_th>>
                        cov_str >>
                        cov_xx >> cov_xy >> cov_xth >>
                        cov_yx >> cov_yy >> cov_yth >>
                        cov_thx >> cov_thy >> cov_thth;
          assert(cov_str.compare("covariance") == 0);
          int n_idx = stoi(node_str.substr(1,node_str.size()-1));
          unsigned char n_chr = (unsigned char)node_str[0];
          Eigen::Matrix3d cov_mat;
          cov_mat<<cov_xx, cov_xy, cov_xth,
                 cov_yx, cov_yy, cov_yth,
                 cov_thx, cov_thy, cov_thth;
          noiseModel::Gaussian::shared_ptr priorNoise = 
            noiseModel::Gaussian::Covariance(cov_mat);
          if (n_chr == rbt_chr || n_chr == lm_chr){
              graph.push_back(PriorFactor<Pose2>(Symbol(n_chr, n_idx), Pose2(m_x, m_y, m_th), priorNoise));
              cout<<"Read prior factor of "
                  <<node_str<<" at ("
                  <<m_x<<","<<m_y<<","<<m_th<<")"
                  <<" and cov_thth("
                  <<cov_thth<<")"
                  <<endl;
          }
          else{
              cerr << "An invalid node_str "<<node_str<<endl;
          }
        }
        else if(factorType.compare("SE2RelativeGaussianLikelihoodFactor") == 0){
          string n1_str;
          string n2_str;
          string cov_str;
          double m_x, m_y, m_th;
          double cov_xx, cov_xy, cov_xth,
                 cov_yx, cov_yy, cov_yth,
                 cov_thx, cov_thy, cov_thth;

          linestream >> n1_str >> n2_str >> m_x >> m_y>> m_th>>
                        cov_str >>
                        cov_xx >> cov_xy >> cov_xth >>
                        cov_yx >> cov_yy >> cov_yth >>
                        cov_thx >> cov_thy >> cov_thth;
          assert(cov_str.compare("covariance") == 0);
          Eigen::Matrix3d cov_mat;
          cov_mat<<cov_xx, cov_xy, cov_xth,
                 cov_yx, cov_yy, cov_yth,
                 cov_thx, cov_thy, cov_thth;
          noiseModel::Gaussian::shared_ptr noise = 
            noiseModel::Gaussian::Covariance(cov_mat);
          int n1_idx = stoi(n1_str.substr(1,n1_str.size()-1));
          int n2_idx = stoi(n2_str.substr(1,n2_str.size()-1));
          unsigned char n1_chr = (unsigned char)n1_str[0];
          unsigned char n2_chr = (unsigned char)n2_str[0];
          if ( (n1_chr == rbt_chr || n1_chr == lm_chr)
               &&
               (n2_chr == rbt_chr || n2_chr == lm_chr)){
            graph.push_back(BetweenFactor<Pose2>(Symbol(n1_chr, n1_idx),Symbol(n2_chr, n2_idx),
                                                  Pose2(m_x, m_y, m_th), noise));                
            cout<<"Read factor between "<<n1_str<<" and "
                <<n2_str<<" with ("
                <<m_x<<","<<m_y<<","<<m_th<<")"
                <<" and cov_thth("
                <<cov_thth<<")"
                <<endl;
          }          
          else{
              cerr << "Invalid node_str "<<n1_str<<", "<<n2_str<<endl;
          }
        }
        else if(factorType.compare("SE2R2RangeGaussianLikelihoodFactor") == 0){
          string n1_str;
          string n2_str;
          double m_r;
          double sigma_r;
          linestream >> n1_str >> n2_str >> m_r
            >> sigma_r;
          int n1_idx = stoi(n1_str.substr(1,n1_str.size()-1));
          int n2_idx = stoi(n2_str.substr(1,n2_str.size()-1));
          noiseModel::Diagonal::shared_ptr noise = noiseModel::Isotropic::Sigma(1, sigma_r);
          unsigned char n1_chr = (unsigned char)n1_str[0];
          unsigned char n2_chr = (unsigned char)n2_str[0];
          if ( (n1_chr == rbt_chr || n1_chr == lm_chr)
               &&
               (n2_chr == rbt_chr || n2_chr == lm_chr)){
            if (n1_chr == rbt_chr){
              graph.push_back(RangeFactor_SE2(Symbol(n1_chr, n1_idx),Symbol(n2_chr, n2_idx),
                                           m_r, noise));              
            }
            else{
              graph.push_back(RangeFactor_SE2(Symbol(n2_chr, n2_idx),Symbol(n1_chr, n1_idx),
                                           m_r, noise));                            
            }
            cout<<"Read range factor between "<<n1_str<<" and "
                <<n2_str<<" with range "
                <<m_r<<endl;
          }          
          else{
              cerr << "Invalid node_str "<<n1_str<<", "<<n2_str<<endl;
          }
        }
        else if(factorType.compare("AmbiguousDataAssociationFactor") == 0){
          vector<string> strVec = {"Observer", "Observed", "Weights", "Binary", "Observation", "Sigma"};
          map<string, int> str2idx;
          vector<int> idxVec;
          vector<string> inputVec;
          string tmp;
          while(linestream>>tmp) inputVec.push_back(tmp);
          for (int kw = 0; kw < strVec.size(); kw++){
		    auto it = find(inputVec.begin(), inputVec.end(), strVec[kw]);
		 
		    // If element was found
		    if (it != inputVec.end())
		    {		     
		        // calculating the index
		        int index = it - inputVec.begin();
		        str2idx[strVec[kw]]=index;
		    }
		    else {
		        // If the element is not
		        // present in the vector
		        cerr << "Missing key words in the input of data association factor." << endl;
		    }                    	
          }

          assert(str2idx["Observer"] == 0);
          assert(str2idx["Observed"] == 2);

          string observer = inputVec[1];

          int observed_num = str2idx["Weights"] - str2idx["Observed"] - 1;
          int weight_num = str2idx["Binary"] - str2idx["Weights"] - 1;
          assert(weight_num == observed_num);

          vector<double> weights;
          vector<string> observed_vars;
	  	  std::vector<Key> all_mixture_keys;
  		  all_mixture_keys.push_back(string2sym(observer));

          for(int idx = 0; idx < weight_num; idx++){
          	weights.push_back(stod(inputVec[str2idx["Weights"]+1+idx]));
          	string observed_v = inputVec[str2idx["Observed"]+1+idx];
          	observed_vars.push_back(observed_v);
          	all_mixture_keys.push_back(string2sym(observed_v));
          }

          string binary_factor = inputVec[str2idx["Binary"]+1];
          if (binary_factor.compare("SE2R2RangeGaussianLikelihoodFactor") == 0){
          	assert(str2idx["Sigma"] - str2idx["Observation"] == 2);
          	double m_r = stod(inputVec[str2idx["Observation"]+1]);
          	double sigma_r = stod(inputVec[str2idx["Sigma"]+1]);
          	std::vector<RangeFactor_SE2> r_factors;
          	for(int i = 0; i < observed_vars.size();i++){
          		r_factors.push_back(constructRangeFactor(observer,
						            observed_vars[i],
						            m_r,
						            sigma_r,
						            rbt_chr));
          	}
          	graph.push_back(maxmixture::MaxMixtureFactor<RangeFactor_SE2>(all_mixture_keys, r_factors, weights));
          	cout<<"Read AmbiguousDataAssociationFactor for range "<<m_r<<" between "<<
          		observer<<" and ";
          	for (auto var: observed_vars){
				cout<<var<<" ";          		
          	}
          	cout<<std::endl;
          }
          else if (binary_factor.compare("SE2RelativeGaussianLikelihoodFactor") == 0){
          	assert(str2idx["Sigma"] - str2idx["Observation"] == 4);
          	double x = stod(inputVec[str2idx["Observation"]+1]);
          	double y = stod(inputVec[str2idx["Observation"]+2]);
          	double th = stod(inputVec[str2idx["Observation"]+3]);

          	Eigen::Matrix3d cov_mat;
            cov_mat<<stod(inputVec[str2idx["Sigma"]+1]), stod(inputVec[str2idx["Sigma"]+2]), stod(inputVec[str2idx["Sigma"]+3]),
                 stod(inputVec[str2idx["Sigma"]+4]), stod(inputVec[str2idx["Sigma"]+5]), stod(inputVec[str2idx["Sigma"]+6]),
                 stod(inputVec[str2idx["Sigma"]+7]), stod(inputVec[str2idx["Sigma"]+8]), stod(inputVec[str2idx["Sigma"]+9]);

          	std::vector<BetweenFactor<Pose2>> p_factors;
          	for(int i = 0; i < observed_vars.size();i++){
          		p_factors.push_back(constructPoseFactor(observer, 
          			observed_vars[i],Pose2(x,y,th),cov_mat));
          	}
          	graph.push_back(maxmixture::MaxMixtureFactor<BetweenFactor<Pose2>>(all_mixture_keys, p_factors, weights));
          	cout<<"Read AmbiguousDataAssociationFactor for pose "<<x<<", "<<y<<", "<<th<<" between "<<
          		observer<<" and ";
          	for (auto var: observed_vars){
				cout<<var<<" ";          		
          	}
          	cout<<std::endl;
          }
          else{
          	cerr<<"Unknown binary factor type."<<endl;
          }
      	}
      	else if(factorType.compare("BinaryFactorWithNullHypo") == 0){
          vector<string> strVec = {"Observer", "Observed", "Weights", "Binary", "Observation", "Sigma","NullSigmaScale"};
          map<string, int> str2idx;
          vector<int> idxVec;
          vector<string> inputVec;
          string tmp;
          while(linestream>>tmp) inputVec.push_back(tmp);
          for (int kw = 0; kw < strVec.size(); kw++){
		    auto it = find(inputVec.begin(), inputVec.end(), strVec[kw]);
		 
		    // If element was found
		    if (it != inputVec.end())
		    {		     
		        // calculating the index
		        int index = it - inputVec.begin();
		        str2idx[strVec[kw]]=index;
		    }
		    else {
		        // If the element is not
		        // present in the vector
		        cerr << "Missing key words in the input of data association factor." << endl;
		    }                    	
          }

          assert(str2idx["Observer"] == 0);
          assert(str2idx["Weights"] == 4);

          string observer = inputVec[1];
          string observed = inputVec[3];
	  	  std::vector<Key> all_mixture_keys;
  		  all_mixture_keys.push_back(string2sym(observer));
  		  all_mixture_keys.push_back(string2sym(observed));

          int weight_num = str2idx["Binary"] - str2idx["Weights"] - 1;

          vector<double> weights;
          for(int idx = 0; idx < weight_num; idx++){
          	weights.push_back(stod(inputVec[str2idx["Weights"]+1+idx]));
          }

          string binary_factor = inputVec[str2idx["Binary"]+1];
          double null_hypo_scale = stod(inputVec[str2idx["NullSigmaScale"]+1]);
          if (binary_factor.compare("SE2R2RangeGaussianLikelihoodFactor") == 0){
          	assert(str2idx["Sigma"] - str2idx["Observation"] == 2);
          	double m_r = stod(inputVec[str2idx["Observation"]+1]);
          	double sigma_r = stod(inputVec[str2idx["Sigma"]+1]);

          	std::vector<RangeFactor_SE2> r_factors;
          	r_factors.push_back(constructRangeFactor(observer,
						            observed,
						            m_r,
						            sigma_r,
						            rbt_chr));
          	r_factors.push_back(constructRangeFactor(observer,
						            observed,
						            m_r,
						            sigma_r * null_hypo_scale,
						            rbt_chr));
          	graph.push_back(maxmixture::MaxMixtureFactor<RangeFactor_SE2>(all_mixture_keys, r_factors, weights));
          	cout<<"Read BinaryFactorWithNullHypo for range "<<m_r<<" between "<<
          		observer<<" and "<<observed<<endl;
          }
          else if (binary_factor.compare("SE2RelativeGaussianLikelihoodFactor") == 0){
          	assert(str2idx["Sigma"] - str2idx["Observation"] == 4);
          	double x = stod(inputVec[str2idx["Observation"]+1]);
          	double y = stod(inputVec[str2idx["Observation"]+2]);
          	double th = stod(inputVec[str2idx["Observation"]+3]);

          	Eigen::Matrix3d cov_mat;
            cov_mat<<stod(inputVec[str2idx["Sigma"]+1]), stod(inputVec[str2idx["Sigma"]+2]), stod(inputVec[str2idx["Sigma"]+3]),
                 stod(inputVec[str2idx["Sigma"]+4]), stod(inputVec[str2idx["Sigma"]+5]), stod(inputVec[str2idx["Sigma"]+6]),
                 stod(inputVec[str2idx["Sigma"]+7]), stod(inputVec[str2idx["Sigma"]+8]), stod(inputVec[str2idx["Sigma"]+9]);
          	std::vector<BetweenFactor<Pose2>> p_factors;
      		p_factors.push_back(constructPoseFactor(observer, 
      			observed,Pose2(x,y,th),cov_mat));
      		p_factors.push_back(constructPoseFactor(observer, 
      			observed,Pose2(x,y,th),cov_mat*null_hypo_scale));
          	graph.push_back(maxmixture::MaxMixtureFactor<BetweenFactor<Pose2>>(all_mixture_keys, p_factors, weights));
          	cout<<"Read BinaryFactorWithNullHypo for pose "<<x<<", "<<y<<", "<<th<<" between "<<
          		observer<<" and "<<observed<<endl;
          }
          else{
          	cerr<<"Unknown binary factor type."<<endl;
          }
        }
        else{
          cerr << "Invalid factor type "<<factorType<<endl;          
        }
        continue;                      
      }
    }
    cout<<"Factor graph file loaded."<<endl;
    cout<<"----------------------------"<<endl;    
  }
  else {
      cerr << "Couldn't open factor graph file for reading.\n";
  }  
}

//return values and indices of factors
vector<pair<Values, vector<int>>> getIncFactors(
const std::vector<Symbol>& raw_sym,
const NonlinearFactorGraph& raw_factors, 
const Values& raw_values, 
int& inc_step,
int gt_init){
  unsigned char rbt_chr = 'X';
  unsigned char lm_chr = 'L';

  vector<pair<Values, vector<int>>> res;
  //robot indices
  vector<int> robot_indices;
  vector<Symbol> initialized_sym;
  vector<int> prior_indices;
  vector<int> like_indices;
  //get robot symbols
  cout<<"Robot symbols are ";
  for (int i = 0; i < raw_sym.size(); i++){
    if (raw_sym[i].chr() == rbt_chr){
      robot_indices.push_back(i);
      cout<<raw_sym[i].chr()<<raw_sym[i].index()<<" ";
    }
  }
  cout<<endl;

  if (inc_step == 0 || inc_step > robot_indices.size()){
    inc_step = robot_indices.size();
    cout<<"Invalid step: reset the incremental step to "
      <<inc_step<<" for a batch solution."<<endl;
  }

  //split prior and likelihood factors
  for (int i = 0; i < raw_factors.size(); i++){
    Key frontKey = raw_factors.at(i)->front();
    Key lastKey = raw_factors.at(i)->back();
    if (frontKey == lastKey){
      prior_indices.push_back(i);  
      cout<<"Found prior factor at "
        <<Symbol(frontKey).chr()
        <<Symbol(frontKey).index()
        <<endl;      
    }
    else{
      like_indices.push_back(i);
      cout<<"Found likelihood factor between ";
      for (auto v_key: raw_factors.at(i)->keys()){
      	cout<<Symbol(v_key).chr()
        <<Symbol(v_key).index()
        <<" ";
      }
      cout<<endl;      
    }
  }

  Values newValues;
  vector<int> newFactors;
  for (int i = 0; i < robot_indices.size(); i++){
    int rbt_idx = robot_indices[i];
    Symbol rbt_sym;
    rbt_sym = raw_sym[rbt_idx];
    newValues.insert(rbt_sym,raw_values.at(rbt_sym));
    initialized_sym.push_back(rbt_sym);
    cout<<"Initialized variable "
      <<rbt_sym.chr()
      <<rbt_sym.index()
      <<"."<<endl;

    for (int idx = 0; idx < prior_indices.size(); idx++){
      int factor_idx = prior_indices[idx];
      Key vKey = raw_factors.at(factor_idx)->front();
      if (vKey == rbt_sym){
        newFactors.push_back(factor_idx);
        cout<<"Push back prior factor at "
          <<rbt_sym.chr()
          <<rbt_sym.index()
          <<endl;               
      }
    }
    for( int j = 0; j < like_indices.size();j++){
      int factor_idx = like_indices[j];
      Key frontKey = raw_factors.at(factor_idx)->front();
      Key lastKey = raw_factors.at(factor_idx)->back();
      Key another_key;
      if(rbt_sym == frontKey){
        another_key = lastKey;
      }
      else if(rbt_sym == lastKey){
        another_key = frontKey;
      }
      else{
        continue;
      }
      if (Symbol(another_key).chr() == rbt_chr){
        if (find(initialized_sym.begin(), initialized_sym.end(), 
          Symbol(another_key)) != initialized_sym.end()){
          newFactors.push_back(factor_idx);
          cout<<"Push back likelihood factor between ";
	      for (auto v_key: raw_factors.at(factor_idx)->keys()){
	      	cout<<Symbol(v_key).chr()
	        <<Symbol(v_key).index()
	        <<" ";
	      }
	      cout<<endl; 
        }
      }
      else if (Symbol(another_key).chr() == lm_chr){
        newFactors.push_back(factor_idx);
        cout<<"Push back likelihood factor between ";
	      for (auto v_key: raw_factors.at(factor_idx)->keys()){
	      	cout<<Symbol(v_key).chr()
	        <<Symbol(v_key).index()
	        <<" ";
	      }
	      cout<<endl; 
        if (find(initialized_sym.begin(), initialized_sym.end(), 
          Symbol(another_key)) == initialized_sym.end()){//uninitialized landmark
          Symbol newSym = Symbol(another_key);
          if(gt_init == 1){
	        newValues.insert(newSym,raw_values.at(newSym));          	
          }
          else{// only works for range measurements
          	Pose2 rbt_pose = raw_values.at<Pose2>(rbt_sym);
          	boost::shared_ptr<RangeFactor_SE2> ptr = boost::dynamic_pointer_cast<RangeFactor_SE2>(raw_factors.at(factor_idx));
	        newValues.insert(newSym,ptr->sampleOnRing(rbt_pose.x(),
	        	rbt_pose.y()));          	
          }
          initialized_sym.push_back(newSym);
          cout<<"Initialized variable "
            <<newSym.chr()
            <<newSym.index()
            <<"."<<endl;
          for (int idx = 0; idx < prior_indices.size(); idx++){
            int factor_idx = prior_indices[idx];
            Key vKey = raw_factors.at(factor_idx)->front();
            if (vKey == another_key){
              newFactors.push_back(factor_idx);
              cout<<"Push back prior factor at "
                <<Symbol(another_key).chr()
                <<Symbol(another_key).index()
                <<endl;               
            }
          }
        }
      }
      else{
        cerr << "Unkonwn symbol:"<<Symbol(another_key).chr()
          <<Symbol(another_key).index()<<endl;
      }
    }
    if( ((i+1)%inc_step) == 0){
      pair<Values, vector<int>> curStep(newValues, newFactors);
      res.push_back(curStep);
      newValues.clear();
      newFactors.clear();
      cout<<"New batch loaded."<<endl;
      cout<<"-----------------"<<endl;
    }
  }
  cout<<"There are "<<res.size()<<" batches of new factors."<<endl;
  cout<<"Batches of incremental factors loaded."<<endl;
  cout<<"----------------------------"<<endl;
  return res;
}

int main(int argc, char *argv[]) {

  //load configuration
  cout<<"You have entered "<<argc<<" arguments."<<endl;

  string input_dir = "in_file";
  string output_dir = "out_file";
  float artificial_prior_sigma = -1.0; //negative means no artificial prior factors
  int gt_init = 0;
  //0 means batch solution
  int inc_step = 1;
  if(argc == 2){
    //using config files
      string configDir = argv[1];
      cout<<"The dir of config file is "<<configDir<<endl;
      readConfigFile(configDir, input_dir, output_dir, inc_step, artificial_prior_sigma,
        gt_init);
  }
  else if (argc == 3){
    //directly type in arguments
        input_dir = argv[1];
        output_dir = argv[2];
   }
   else if(argc == 4){
        input_dir = argv[1];
        output_dir = argv[2];
        inc_step = stoi(argv[3]);
   }
   else if(argc == 5){
        input_dir = argv[1];
        output_dir = argv[2];
        inc_step = stoi(argv[3]);
        artificial_prior_sigma = stof(argv[4]);
        }
   else{
        input_dir = argv[1];
        output_dir = argv[2];
        inc_step = stoi(argv[3]);
        artificial_prior_sigma = stof(argv[4]);
        gt_init = stoi(argv[5]);
  }

	// Creating a directory
    // Creating a directory
    if (mkdir(output_dir.c_str(), 0777) == -1)
        cerr << "Error :  " << strerror(errno) << endl;
  
    else
        cout << "Directory created"<<endl;

  //parse input factor graph
  std::vector<Symbol> raw_sym;
  NonlinearFactorGraph raw_factors;
  Values raw_values;
  readFactorGraph(input_dir, raw_sym, raw_factors, raw_values, artificial_prior_sigma);

  if(inc_step<1){
    inc_step = raw_sym.size();
  }
  //start solving the problem incrementally
  vector<pair<Values, vector<int>>> factors_vec;
  factors_vec = getIncFactors(raw_sym, raw_factors, raw_values, inc_step, gt_init);

  //start solving the problem incrementally
  NonlinearFactorGraph inc_factor_graph;
  Values inc_values;

  //using ISAM2
  ISAM2 isam;
  NonlinearFactorGraph newFactors;
  Values newInitial;


  vector<double> time_list;
  for (int i = 0; i < factors_vec.size(); i++){    
    auto start = chrono::high_resolution_clock::now();   
    // unsync the I/O of C and C++. 
    ios_base::sync_with_stdio(false); 
  	
  	newFactors = NonlinearFactorGraph();
    newInitial = Values();

    for (int j = 0; j < factors_vec[i].second.size(); j++){
      int factor_idx = factors_vec[i].second[j];
      inc_factor_graph.add(raw_factors.at(factor_idx));
      newFactors.add(raw_factors.at(factor_idx));
    }
    inc_values.insert(factors_vec[i].first);
    newInitial.insert(factors_vec[i].first);

    // isam.update(newFactors, newInitial);
    // Values result = isam.calculateEstimate();
    LevenbergMarquardtOptimizer batchOptimizer(inc_factor_graph, inc_values);
    Values result;
    result = batchOptimizer.optimize();

    auto end = chrono::high_resolution_clock::now(); 
  
    // Calculating total time taken by the program. 
    double time_taken =  
      chrono::duration_cast<chrono::nanoseconds>(end - start).count(); 
  
    time_taken *= 1e-9; 
    cout<<"Step "<<i<<" results:"<<endl;
    result.print();

    //vector for writing files
    map<Key, vector<double>> out_data;
    //int batch_dim = 2 * result.keys().size();
    Eigen::MatrixXd jointCov;// = Eigen::MatrixXd::Zero(batch_dim, batch_dim);

    Values pose2_res = result.filter<Pose2>(true_fcn);
    Values point2_res = result.filter<Point2>(true_fcn);

    for (Key key_i: pose2_res.keys()){
        out_data[key_i].push_back(result.at<Pose2>(key_i).x());
        out_data[key_i].push_back(result.at<Pose2>(key_i).y()); 
        out_data[key_i].push_back(result.at<Pose2>(key_i).theta());   
    }
    for (Key key_i: point2_res.keys()){
        out_data[key_i].push_back(result.at<Point2>(key_i).x());
        out_data[key_i].push_back(result.at<Point2>(key_i).y());        
    }

    try{
      Marginals marginals(inc_factor_graph, result, Marginals::QR);
      jointCov = marginals.jointMarginalCovariance(result.keys()).fullMatrix();  

      for (Key key_i: pose2_res.keys()){
        Symbol sym_i(key_i);
        Eigen::Matrix3d mat = marginals.marginalCovariance(sym_i);
        cout << sym_i.chr()<<sym_i.index()
          <<" covariance:\n" << mat << endl;
        out_data[key_i].push_back(mat(0,0));
        out_data[key_i].push_back(mat(0,1));
        out_data[key_i].push_back(mat(0,2));
        out_data[key_i].push_back(mat(1,0));
        out_data[key_i].push_back(mat(1,1));
        out_data[key_i].push_back(mat(1,2));
        out_data[key_i].push_back(mat(2,0));
        out_data[key_i].push_back(mat(2,1));
        out_data[key_i].push_back(mat(2,2));
      }
      for (Key key_i: point2_res.keys()){
        Symbol sym_i(key_i);
        Eigen::Matrix2d mat = marginals.marginalCovariance(sym_i);
        cout << sym_i.chr()<<sym_i.index()
          <<" covariance:\n" << mat << endl;
        out_data[key_i].push_back(mat(0,0));
        out_data[key_i].push_back(mat(0,1));
        out_data[key_i].push_back(mat(1,0));
        out_data[key_i].push_back(mat(1,1));
      }
    }
    catch(...){
      cout<<"Couldn't compute marginals at this step."<<endl;
    }
    cout << "Time taken by program is : "
     << time_taken << setprecision(9); 
    cout << " sec" << endl;
    cout<<"----------------------------"<<endl;
    time_list.push_back(time_taken);

    //write out results
    string file_name = output_dir + "/step_"+ to_string(i)+"_marginal";
    string file_name2 = output_dir + "/step_"+ to_string(i)+"_joint";
    string file_ordering = output_dir + "/step"+ to_string(i)+"_ordering";
    ofstream out_file(file_name);
    ofstream out_file2(file_name2);
    ofstream out_ordering(file_ordering);
    if (out_file.is_open() && out_file2.is_open() && out_ordering.is_open())
    {
      // out_file<<"Variable "<<"mean_x "<<"mean_y "<<"cov_xx "
      //   <<"cov_xy "<<"cov_yx "<<"cov_yy\n";
      for (Key key_i: result.keys()){
        Symbol sym_i(key_i);
        for (double v : out_data[key_i]){
          out_file<<v<<" ";
        }
        out_file<<"\n";
        out_ordering<<sym_i.chr()<<sym_i.index()<<" ";
      }
      out_ordering.close();
      out_file.close();
      out_file2<<jointCov;
      out_file2.close();
    }
  }
  string time_file_name = output_dir + "/timing";
  ofstream time_file(time_file_name);
  if(time_file.is_open()){
    for (double t: time_list){
      time_file<<t<<" ";      
    }
    time_file.close();
  }
  return 0;
}