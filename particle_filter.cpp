#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;
static default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	num_particles = 100;
	
	weights.resize(num_particles);
	particles.resize(num_particles);

    // Assigning normal distributions for GPS sensor noise
	normal_distribution<double> dist_x(0, std[0]);
	normal_distribution<double> dist_y(0, std[1]);
	normal_distribution<double> dist_theta(0, std[2]);

	for (int i=0; i<num_particles; i++)
	{
		weights[i] = 1.0;

		particles[i].id = i;
		particles[i].x = x + dist_x(gen);
		particles[i].y = y + dist_y(gen);
		particles[i].theta = theta + dist_theta(gen);
		particles[i].weight = 1.0;
  	}

	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	// Assigning normal distributions for sensor noise
	normal_distribution<double> dist_x(0, std_pos[0]);
	normal_distribution<double> dist_y(0, std_pos[1]);
	normal_distribution<double> dist_theta(0, std_pos[2]);

	for (int i=0; i<num_particles; i++) {

		double theta = particles[i].theta;
		        
        // If the car is going in a straight line, i.e., yaw rate close to zero
		if (fabs(yaw_rate) < 0.001)
	    {
	      particles[i].x += velocity * delta_t * cos(theta) + dist_x(gen);
	      particles[i].y += velocity * delta_t * sin(theta) + dist_y(gen);
	    }
	    else
	    {
	      particles[i].x += (velocity * (sin(theta + yaw_rate * delta_t) - sin(theta)) / yaw_rate) + dist_x(gen);
	      particles[i].y += (velocity * (cos(theta) - cos(theta + yaw_rate * delta_t)) / yaw_rate) + dist_y(gen);
	      particles[i].theta += (yaw_rate * delta_t) + dist_theta(gen);
	    }
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

	vector<LandmarkObs> landmarks;
	  for (int i=0; i<observations.size(); i++)
	  {
	    int map_index;
	    double dist_min = numeric_limits<double>::max();
	    for (int j=0; j<predicted.size(); j++)
	    {
	      double delta_x = predicted[j].x - observations[i].x;
	      double delta_y = predicted[j].y - observations[i].y;
	      double delta = (delta_x * delta_x) + (delta_y * delta_y); 
	      if (delta < dist_min) {
	      	map_index = j;
		    dist_min = delta;
	      } 
	    }
	    LandmarkObs result = {map_index, observations[i].x, observations[i].y};
	    landmarks.push_back(result);
	  }
	observations = landmarks;
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
	
	vector<double> updated_weights;
    
    // iterate over all particles to update their weights
	for (int i=0; i<num_particles; i++)
	{
		double p_x = particles[i].x;
	    double p_y = particles[i].y;
	    double p_theta = particles[i].theta;

	    // Find out the map landmarks that are within the range of the sensor 
	    vector<LandmarkObs> map_landmarks_in_sensor_range;
	    for (int j=0; j<map_landmarks.landmark_list.size(); j++)
	    {
	      int index = map_landmarks.landmark_list[j].id_i;
	      double m_x = map_landmarks.landmark_list[j].x_f;
	      double m_y = map_landmarks.landmark_list[j].y_f;
	      if (dist(p_x, p_y, m_x, m_y) < sensor_range)
	      { 
	      	LandmarkObs result = {index, m_x, m_y};
		    map_landmarks_in_sensor_range.push_back(result); 
	      }
	    }

	   // Transforming observations from vehicle co-ordinates to map co-ordinates
	    vector<LandmarkObs> transformed_observations;
	    for (int j=0; j<observations.size(); j++)
	    {
	      int index = observations[j].id;
	      double o_x = observations[j].x;
	      double o_y = observations[j].y;
	      double m_x = p_x + o_x * cos(p_theta) - o_y * sin(p_theta);
	      double m_y = p_y + o_x * sin(p_theta) + o_y * cos(p_theta); 
	      LandmarkObs result = {index, m_x, m_y};
	      transformed_observations.push_back(result); 
	    }

	    // Associate the transformed observations to the closest map landmark using the data-association class defined above 
	    dataAssociation(map_landmarks_in_sensor_range, transformed_observations);

	    // Compute the updated weights based on observations in map coordinates and coordinates of the nearest landmarks
	    double updated_weight = 1.0; 
	    for (int j=0; j<transformed_observations.size(); j++)
	    {
	      double o_x = transformed_observations[j].x;
	      double o_y = transformed_observations[j].y;
	      int map_index = transformed_observations[j].id;
	      double m_x = map_landmarks_in_sensor_range[map_index].x;
	      double m_y = map_landmarks_in_sensor_range[map_index].y;
	      
	      // Applying Multivariate-Gaussian probability to calculate updated weight
	      double a = (o_x - m_x) * (o_x - m_x) / (2.0 * std_landmark[0] * std_landmark[0]);
	      double b = (o_y - m_y) * (o_y - m_y) / (2.0 * std_landmark[1] * std_landmark[1]);
	      updated_weight *= (exp(-1.0 * (a + b))) / (2.0 * M_PI * std_landmark[0] * std_landmark[1]);
	    } 
	    particles[i].weight = updated_weight;
	    weights[i] = updated_weight;
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	double total_weight = 0.0;
	  for (int i=0; i<particles.size(); i++)
	  {
	    total_weight += particles[i].weight;
	  }
	  for (int i=0; i<particles.size(); i++)
	  {
	    particles[i].weight /= total_weight;
	    weights[i] /= total_weight;	    
	  }

	  // Calculate the maximum of these weights
	  double max_weight = *max_element(weights.begin(), weights.end());

	// Resampling
	  double beta = 0.0;
	  
	  //Generating random index for resampling wheel
	  uniform_int_distribution<int> uniintdist(0, num_particles-1);
      int index = uniintdist(gen);

      // uniform random distribution [0.0, max_weight)
      uniform_real_distribution<double> weight_dist(0.0, max_weight);

	  vector<Particle> new_particles;
	  
	  for (int i=0; i<num_particles; i++)
	  {
	    beta += weight_dist(gen) * 2.0;
	    while (weights[index] < beta)
	    {
	      beta -= weights[index];
	      index = (index + 1) % num_particles;
	    }
	    new_particles.push_back(particles[index]);
	  }
	  particles = new_particles;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
