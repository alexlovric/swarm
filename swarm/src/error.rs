use core::fmt;
pub type Result<T> = core::result::Result<T, SwarmError>;

#[derive(Debug)]
pub enum SwarmError {
    InvalidFormat(String),
    FailedFunctionCall(String),
    FailedToGetSolution,
    FailedToFindValidPoint,
    FailedToMeetCondition(String),
}

impl fmt::Display for SwarmError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            SwarmError::InvalidFormat(msg) => {
                write!(f, "🔴 Optimisation invalid format error!: {msg}")
            }
            SwarmError::FailedFunctionCall(msg) => write!(f, "🔴 Function call failed!: {msg}"),
            SwarmError::FailedToGetSolution => {
                write!(f, "🔴 Optimisation failed to find solution!")
            }
            SwarmError::FailedToFindValidPoint => {
                write!(f, "🔴 Optimisation failed to find valid point!")
            }
            SwarmError::FailedToMeetCondition(msg) => {
                write!(f, "🔴 Optimisation failed to meet condition!: {msg}")
            }
        }
    }
}
