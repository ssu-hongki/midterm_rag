import { motion } from 'framer-motion';
import { ExternalLink } from 'lucide-react';
import { JobRecommendation } from '../data/jobRecommendations';

interface JobRecommendationCardProps {
  job: JobRecommendation;
  onSelect: (job: JobRecommendation) => void;
  index: number;
}

export function JobRecommendationCard({ job, onSelect, index }: JobRecommendationCardProps) {
  return (
    <motion.button
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.05 }}
      whileHover={{ y: -4 }}
      onClick={() => onSelect(job)}
      className="group text-left w-full"
    >
      <div className="bg-white rounded-lg border border-gray-200 hover:border-gray-400 transition-all shadow-sm hover:shadow-md overflow-hidden h-full cursor-pointer">
        {/* Header */}
        <div className={`bg-gradient-to-r ${job.color} p-4 text-white`}>
          <div className="flex items-center justify-between mb-3">
            <div className="text-sm font-medium opacity-80">{job.category}</div>
            {job.url !== '#' && (
              <a
                href={job.url}
                target="_blank"
                rel="noopener noreferrer"
                onClick={(e) => {
                  e.stopPropagation();
                }}
                className="p-1.5 bg-white/10 rounded-lg hover:bg-white/20 transition-colors backdrop-blur-sm"
              >
                <ExternalLink className="w-4 h-4" />
              </a>
            )}
          </div>
          <div className="space-y-1">
            <h3 className="font-semibold text-base leading-tight">{job.position}</h3>
            <p className="text-sm text-white/80">{job.company}</p>
          </div>
        </div>

        {/* Content */}
        <div className="p-4 space-y-3">
          {/* Skills */}
          <div>
            <p className="text-xs font-semibold text-gray-500 mb-2">주요 기술</p>
            <div className="flex flex-wrap gap-1.5">
              {job.skills.slice(0, 4).map((skill, i) => (
                <span
                  key={i}
                  className="px-2 py-1 bg-gray-100 text-gray-700 rounded-md text-xs font-medium"
                >
                  {skill}
                </span>
              ))}
            </div>
          </div>

          {/* Requirements */}
          <div>
            <p className="text-xs font-semibold text-gray-500 mb-2">자격 요건</p>
            <ul className="space-y-1">
              {job.requirements.slice(0, 2).map((req, i) => (
                <li key={i} className="text-xs text-gray-600 flex items-start gap-1">
                  <span className="text-primary-500 mt-0.5">•</span>
                  <span className="flex-1">{req}</span>
                </li>
              ))}
            </ul>
          </div>

        </div>
      </div>
    </motion.button>
  );
}

