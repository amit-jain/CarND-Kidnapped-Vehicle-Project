/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>
#include <map>
#include "particle_filter.h"

using namespace std;

// declare a random engine to be used across multiple and various method calls
static default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

	// 2 * particle per m till the sensor_range
	num_particles = 2 * 50;

	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_psi(theta, std[2]);

	for (int i = 0; i < num_particles; i++) {
		Particle particle;
		particle.id = i;
		particle.x = dist_x(gen);
		particle.y = dist_y(gen);
		particle.theta = dist_psi(gen);
		particle.weight = 1.0;

		particles.push_back(particle);
	}

	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	normal_distribution<double> dist_x(0, std_pos[0]);
	normal_distribution<double> dist_y(0, std_pos[1]);
	normal_distribution<double> dist_psi(0, std_pos[2]);

	for (int i = 0; i < num_particles; i++) {
		// calculate new state
		if (fabs(yaw_rate) < 0.00001) {
			particles[i].x += velocity * delta_t * cos(particles[i].theta) + dist_x(gen);
			particles[i].y += velocity * delta_t * sin(particles[i].theta) + dist_y(gen);
		} else {
			particles[i].x += velocity / yaw_rate * (sin(particles[i].theta + yaw_rate * delta_t) - sin(particles[i].theta)) +
												dist_x(gen);
			particles[i].y += velocity / yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate * delta_t)) +
												dist_y(gen);
      particles[i].theta += yaw_rate * delta_t + dist_psi(gen);
		}
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
	for (int i = 0; i < observations.size(); i++) {
		double min_dist = numeric_limits<double>::max();
		LandmarkObs minLandmark;

		for (int j = 0; j < predicted.size(); j++) {
			double distance = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);

			if (distance < min_dist) {
				min_dist = distance;
				minLandmark = predicted[j];
			}
		}
		observations[i].id = minLandmark.id;
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

  for (int i = 0; i < num_particles; i++) {
		Particle particle = particles[i];
		double particle_x = particle.x;
		double particle_y = particle.y;
		double particle_theta = particle.theta;

		// vector to hold the map landmark locations predicted within sensor range of the particle
		std::vector<LandmarkObs> predictions;
		// Map to hold id to predicted landmark mapping
		std::map<int, LandmarkObs> idToPredictedMap;

		for (int j = 0; j < map_landmarks.landmark_list.size(); j++) {
			LandmarkObs predicted;
			predicted.id = map_landmarks.landmark_list[j].id_i;
			predicted.x = map_landmarks.landmark_list[j].x_f;
			predicted.y = map_landmarks.landmark_list[j].y_f;

			if (fabs(predicted.x - particle_x) <= sensor_range && fabs(predicted.y - particle_y) <= sensor_range) {
				predictions.push_back(predicted);
				idToPredictedMap[predicted.id] = predicted;
			}
		}

		// convert to map co-ordinates w.r.t particle
		std::vector<LandmarkObs> transformedObs;
		for (int j = 0; j < observations.size(); j++) {
			LandmarkObs ob = observations[j];
			double xmap = ob.x * cos(particle_theta) - ob.y * sin(particle_theta) + particle_x;
			double ymap = ob.x * sin(particle_theta) + ob.y * cos(particle_theta) + particle_y;
			transformedObs.push_back(LandmarkObs{ob.id, xmap, ymap});
		}

		// associate the nearest landmark to the observations
		dataAssociation(predictions, transformedObs);

		// track the associated landmarks and the mapped sensor measurements
		vector<int> assns;
		vector<double> sense_x;
		vector<double> sense_y;
		for (int j = 0; j < transformedObs.size(); j++) {
			LandmarkObs ob = transformedObs[j];
			assns.push_back(ob.id);
			sense_x.push_back(ob.x);
			sense_y.push_back(ob.y);
		}
		SetAssociations(particle, assns, sense_x, sense_y);

		particles[i].weight = 1.0;
		for (int j = 0; j < transformedObs.size(); j++) {
			double obs_x, obs_y, pred_x, pred_y;

			obs_x = transformedObs[j].x;
			obs_y = transformedObs[j].y;
			int assn_id = transformedObs[j].id;

      // Get the predicted x,y for the association landmark id
			pred_x = idToPredictedMap[assn_id].x;
			pred_y = idToPredictedMap[assn_id].y;

			// calculate weight for this observation with multivariate Gaussian
			double s_x = std_landmark[0];
			double s_y = std_landmark[1];
			double sx2 = 2 * pow(s_x, 2);
			double sy2 = 2 * pow(s_y, 2);

			double obs_w = ( 1.0 / (2.0 * M_PI * s_x * s_y)) * exp(-(pow(pred_x - obs_x, 2)/sx2 +
							pow(pred_y - obs_y, 2)/sy2));

			// multiply all observation weights to get a final weight
			particles[i].weight *= obs_w;
    }
	}
}

void ParticleFilter::resample() {
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	// initialize the weights again
  vector<double> weights;
  for (int i = 0; i < num_particles; i++) {
    weights.push_back(particles[i].weight);
  }

	vector<Particle> new_particles;

	// generate random starting index for resampling wheel
	uniform_int_distribution<int> intdist(0, num_particles-1);
	int index = intdist(gen);

	// get max weight
	double max_weight = *max_element(weights.begin(), weights.end());

	// uniform random distribution [0.0, max_weight)
	uniform_real_distribution<double> realdist(0.0, max_weight);

	double beta = 0.0;
	for (int i = 0; i < num_particles; i++) {
		beta += realdist(gen) * 2.0;

		while (beta > weights[index]) {
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
